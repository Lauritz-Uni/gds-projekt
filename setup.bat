@echo off
echo STARTING SETUP

echo COMPILING RUST
cd rust-preprocess
cargo build --release
cd ..

ECHO DONE