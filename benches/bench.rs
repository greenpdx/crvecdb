use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use crvecdb::{Index, DistanceMetric};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for dim in [128, 384, 768] {
        group.bench_with_input(BenchmarkId::new("dimension", dim), &dim, |b, &dim| {
            b.iter(|| {
                let mut index = Index::builder(dim)
                    .metric(DistanceMetric::Euclidean)
                    .m(16)
                    .ef_construction(100)
                    .capacity(1000)
                    .build()
                    .unwrap();

                let vector: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
                for i in 0..100 {
                    index.insert(i, &vector).unwrap();
                }
                black_box(index)
            });
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for size in [1000, 10000] {
        // Build index once
        let mut index = Index::builder(128)
            .metric(DistanceMetric::Euclidean)
            .m(16)
            .ef_construction(200)
            .capacity(size + 100)
            .build()
            .unwrap();

        for i in 0..size {
            let vector: Vec<f32> = (0..128).map(|j| ((i * 17 + j) % 100) as f32 / 100.0).collect();
            index.insert(i as u64, &vector).unwrap();
        }

        let query: Vec<f32> = vec![0.5; 128];

        group.bench_with_input(BenchmarkId::new("size", size), &index, |b, index| {
            b.iter(|| {
                black_box(index.search(&query, 10).unwrap())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_search);
criterion_main!(benches);
