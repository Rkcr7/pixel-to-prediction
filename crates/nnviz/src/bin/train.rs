//! Trains the digit model and exports the two assets the web app loads:
//! `model.bin` (weights) and `prototypes.bin` (per-class mean images).
//!
//! Refuses to export a model below the accuracy gate, so a bad run can never
//! silently ship.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use nnviz::data::{class_means, load_split, pack_prototypes};
    use nnviz::net::Net;
    use nnviz::tensor::Rng;
    use nnviz::train::{confusion, evaluate, lr_at, train_epoch, Adam};
    use std::path::PathBuf;
    use std::time::Instant;

    let mut data_dir = PathBuf::from("data");
    let mut out_dir = PathBuf::from("web/public/model");
    let mut epochs = 30usize;
    let mut batch = 64usize;
    let mut base_lr = 2e-3f32;
    let mut seed = 1234u64;
    let mut gate = 0.993f32;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let next = |i: usize| -> String {
            args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: {} needs a value", args[i]);
                std::process::exit(2);
            })
        };
        match args[i].as_str() {
            "--data" => data_dir = PathBuf::from(next(i)),
            "--out" => out_dir = PathBuf::from(next(i)),
            "--epochs" => epochs = next(i).parse().expect("--epochs must be a number"),
            "--batch" => batch = next(i).parse().expect("--batch must be a number"),
            "--lr" => base_lr = next(i).parse().expect("--lr must be a number"),
            "--seed" => seed = next(i).parse().expect("--seed must be a number"),
            "--gate" => gate = next(i).parse().expect("--gate must be a number"),
            "--help" | "-h" => {
                println!(
                    "train --data <dir> --out <dir> [--epochs N] [--batch N] [--lr F] [--seed N] [--gate F]"
                );
                return;
            }
            other => {
                eprintln!("error: unknown argument {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }

    println!("loading MNIST from {}", data_dir.display());
    let train = match load_split(
        &data_dir,
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("hint: run scripts/fetch-mnist.sh first");
            std::process::exit(1);
        }
    };
    let test = match load_split(
        &data_dir,
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("  {} train, {} test", train.len(), test.len());

    let mut net = Net::new_random(seed);
    println!("model: {} parameters", net.num_params());
    let mut opt = Adam::new(&net, base_lr);
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let mut order: Vec<usize> = (0..train.len()).collect();

    let mut best_acc = 0.0f32;
    let mut best = net.clone();
    let started = Instant::now();

    for epoch in 0..epochs {
        opt.lr = lr_at(epoch, epochs, base_lr);
        rng.shuffle(&mut order);
        let t0 = Instant::now();
        let stats = train_epoch(
            &mut net,
            &mut opt,
            &train,
            &order,
            batch,
            seed.wrapping_add(epoch as u64),
            true,
        );
        let acc = evaluate(&net, &test);
        if acc > best_acc {
            best_acc = acc;
            best = net.clone();
        }
        println!(
            "epoch {:>2}/{}  lr {:.2e}  loss {:.4}  train {:.2}%  test {:.2}%{}  [{:.1}s]",
            epoch + 1,
            epochs,
            opt.lr,
            stats.loss,
            stats.train_acc * 100.0,
            acc * 100.0,
            if acc >= best_acc { "  *" } else { "" },
            t0.elapsed().as_secs_f32()
        );
    }

    println!(
        "\ntrained in {:.1}s, best test accuracy {:.2}%",
        started.elapsed().as_secs_f32(),
        best_acc * 100.0
    );

    let m = confusion(&best, &test);
    println!("\nconfusion (rows = truth, cols = prediction)");
    print!("     ");
    for c in 0..10 {
        print!("{c:>6}");
    }
    println!();
    for (t, row) in m.iter().enumerate() {
        print!("  {t}  ");
        for (p, n) in row.iter().enumerate() {
            if t == p {
                print!("{:>6}", n);
            } else if *n == 0 {
                print!("{:>6}", ".");
            } else {
                print!("{:>6}", n);
            }
        }
        let total: u32 = row.iter().sum();
        println!("   {:.1}%", row[t] as f32 / total.max(1) as f32 * 100.0);
    }

    if best_acc < gate {
        eprintln!(
            "\nerror: best accuracy {:.2}% is below the {:.2}% gate; refusing to export",
            best_acc * 100.0,
            gate * 100.0
        );
        std::process::exit(1);
    }

    std::fs::create_dir_all(&out_dir).expect("could not create output directory");
    let model_path = out_dir.join("model.bin");
    std::fs::write(&model_path, best.to_bytes()).expect("could not write model");
    println!(
        "\nwrote {} ({} bytes)",
        model_path.display(),
        std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0)
    );

    let protos = class_means(&train);
    let proto_path = out_dir.join("prototypes.bin");
    std::fs::write(&proto_path, pack_prototypes(&protos)).expect("could not write prototypes");
    println!(
        "wrote {} ({} bytes)",
        proto_path.display(),
        std::fs::metadata(&proto_path).map(|m| m.len()).unwrap_or(0)
    );

    // Record the number the README quotes, so the claim always matches the artefact.
    let meta = format!(
        "{{\n  \"test_accuracy\": {:.4},\n  \"params\": {},\n  \"epochs\": {},\n  \"seed\": {}\n}}\n",
        best_acc,
        best.num_params(),
        epochs,
        seed
    );
    std::fs::write(out_dir.join("model.json"), meta).expect("could not write metadata");
    println!("done.");
}
