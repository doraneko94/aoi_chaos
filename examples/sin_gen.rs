use aoi_chaos::*;

use ndarray::*;

fn main() {
    let x = arr1(&[0.0]);
    let v = arr2(&[[1.0]]);
    let f = arr2(&[[1.0]]);
    let g = arr2(&[[1.0]]);
    let h = arr2(&[[1.0]]);
    let q = arr2(&[[1.0]]);
    let r = arr2(&[[10.0]]);

    let init_member = KFMember { x, v, f, g, h, q, r };
    let mut kf_filter = KF::new(&init_member, "./sin_gen/sin_observation.dat");
    kf_filter.run("KF_DA.dat");
}