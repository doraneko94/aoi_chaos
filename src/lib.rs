use ndarray::*;
use ndarray_linalg::*;

use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::io::prelude::*;

#[derive(Clone)]
pub struct KFMember {
    pub x: Array1<f64>,

    pub v: Array2<f64>,
    pub f: Array2<f64>,
    pub g: Array2<f64>,
    pub h: Array2<f64>,
    pub q: Array2<f64>,
    pub r: Array2<f64>,
}

pub struct KF {
    member: KFMember,
    observation_data: Vec<(f64, Array1<f64>)>,
    observation_span: f64,
    simulation_time_length: f64,

}

impl KF {
    const DT: f64 = 0.01;

    pub fn new(init_memmber: &KFMember, observation_file_path: &str) -> Self {
        let member = init_memmber.clone();
        let mut ret = Self { 
            member, 
            observation_data: Vec::new(), 
            observation_span: 0.0, 
            simulation_time_length: 0.0 
        };
        ret.read_observation_file(observation_file_path);
        ret
    }

    pub fn run(&mut self, out_path: &str) {
        let mut out = match fs::File::create(out_path) {
            Ok(inner) => BufWriter::new(inner),
            Err(_) => panic!("cannot create: {}", out_path),
        };
        let mut predict_time_length = 0.0;
        let mut observation_index = 0;
        for loop_index in 0..(self.simulation_time_length / Self::DT) as usize {
            self.predict();
            if predict_time_length >= self.observation_span {
                predict_time_length = 0.0;
                let observation_datum = &self.observation_data[observation_index];
                let observation_time = observation_datum.0;
                if (observation_time as f64 - loop_index as f64 * Self::DT).abs() > Self::DT / 10.0 {
                    panic!("inconsistent observation datum");
                }
                self.filter(observation_index);
                observation_index += 1;
            }
            predict_time_length += Self::DT;
            self.write_x(&mut out, loop_index as f64 * Self::DT);
        }
    }

    fn predict(&mut self) {
        let mut next_member = self.member.clone();
        next_member.x = self.member.f.dot(&self.member.x);
        next_member.v = self.member.f.dot(&self.member.v).dot(&self.member.f.t())
                        + self.member.g.dot(&self.member.q).dot(&self.member.g.t());
        self.member = next_member;
    }

    fn filter(&mut self, observation_index: usize) {
        let y = &self.observation_data[observation_index].1;
        /*
        println!("{:?}", &self.member.h);
        println!("{:?}", &self.member.v);
        println!("{:?}", &self.member.h.dot(&self.member.v));
        println!("{:?}", &self.member.h.dot(&self.member.v).dot(&self.member.h.t()));
        println!("{:?}", &self.member.r);
        println!("{:?}", &self.member.h.dot(&self.member.v).dot(&self.member.h.t()) + &self.member.r);
        */
        let k = self.member.v.dot(&self.member.h.t()).dot(
            &(&self.member.h.dot(&self.member.v).dot(&self.member.h.t()) + &self.member.r).inv().unwrap()
        );
        let mut next_member = self.member.clone();
        next_member.x = &self.member.x + &k.dot(&(y - &self.member.h.dot(&self.member.x)));
        next_member.v = &self.member.v - &k.dot(&self.member.h.dot(&self.member.v));
        self.member = next_member;
        
        // panic!();
    }

    fn read_observation_file(&mut self, observation_file_path: &str) {
        let obs_file = match fs::File::open(observation_file_path) {
            Ok(inner) => BufReader::new(inner),
            Err(_) => panic!("cannot open: {}", observation_file_path),
        };
        for result in obs_file.lines() {
            match result {
                Ok(line) => {
                    let words: Vec<&str> = line.split(" ").collect();
                    if words[0] == "time_span" {
                        self.observation_span = words[1].parse().ok().unwrap();
                    } else if words[0] == "simulation_time" {
                        self.simulation_time_length = words[1].parse().ok().unwrap();
                    } else {
                        let time: f64 = words[0].parse().ok().unwrap();
                        let mut obs_data = Vec::<f64>::new();
                        for obs_index in 1..words.len() {
                            obs_data.push(words[obs_index].parse().ok().unwrap());
                        }
                        let y = arr1(&obs_data);
                        self.observation_data.push((time, y));
                    }
                }
                Err(_) => panic!(),
            };

        }
    }

    fn write_x(&self, out: &mut BufWriter<fs::File>, time: f64) {
        let _ = write!(out, "{} ", time);
        for xi in self.member.x.iter() {
            let _ = write!(out, "{}", xi);
        }
        let _ = write!(out, "\n");
    }
}