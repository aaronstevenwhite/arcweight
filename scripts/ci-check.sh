#!/bin/bash
# CI simulation script to catch issues locally before pushing
# This script runs the same checks as our GitHub CI

set -e  # Exit on any error

echo "🔍 Running CI simulation checks..."

echo ""
echo "📋 1. Running tests (all features)..."
cargo test --all-features --verbose

echo ""
echo "📋 2. Running tests (no default features)..."
cargo test --no-default-features --verbose

echo ""
echo "📋 3. Running Clippy (all features)..."
cargo clippy --all-features --all-targets -- -D warnings

echo ""
echo "📋 4. Running Clippy (no default features)..."
cargo clippy --no-default-features -- -D warnings

echo ""
echo "📋 5. Checking code formatting..."
cargo fmt --all -- --check

echo ""
echo "📋 6. Building documentation..."
cargo doc --all-features --no-deps --document-private-items

echo ""
echo "📋 7. Compiling benchmarks..."
cargo bench --no-run --verbose

echo ""
echo "📋 8. Running examples..."
cargo run --example edit_distance
cargo run --example string_alignment
cargo run --example morphological_analyzer
cargo run --example phonological_rules
cargo run --example pronunciation_lexicon
cargo run --example transliteration
cargo run --example number_date_normalizer
cargo run --example spell_checking

echo ""
echo "✅ All CI checks passed! Safe to push to GitHub."