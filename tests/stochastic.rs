use lmm::stochastic::{StochasticEnhancer, SynonymBank};

#[test]
fn synonym_bank_has_curated_entries() {
    let bank = SynonymBank::new();
    assert!(bank.curated_count() > 50);
}

#[test]
fn synonym_bank_has_wordlist() {
    let bank = SynonymBank::new();
    let has_dictionary = [
        "/usr/share/dict/american-english",
        "/usr/share/dict/english",
        "/usr/share/dict/words",
        "/usr/dict/words",
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists());

    if has_dictionary {
        assert!(bank.wordlist_len() > 0);
    } else {
        assert_eq!(bank.wordlist_len(), 0);
    }
}

#[test]
fn enhancer_changes_text_when_probability_one() {
    let enhancer = StochasticEnhancer::new(1.0);
    let input = "Mathematics enables the representation of reality through elegant equations.";
    let run1 = enhancer.enhance(input);
    let run2 = enhancer.enhance(input);
    assert_ne!(run1, run2, "Two runs at probability=1.0 should differ");
}

#[test]
fn enhancer_preserves_text_when_probability_zero() {
    let enhancer = StochasticEnhancer::new(0.0);
    let input = "The ancient Egyptians built the pyramids using mathematical knowledge.";
    let result = enhancer.enhance(input);
    assert_eq!(result, input);
}

#[test]
fn enhancer_preserves_sentence_count() {
    let enhancer = StochasticEnhancer::new(0.8);
    let input =
        "Equations govern reality. Mathematics reveals truth. Physics shapes our understanding.";
    let result = enhancer.enhance(input);
    let input_sentences = input.matches('.').count();
    let output_sentences = result.matches('.').count();
    assert_eq!(input_sentences, output_sentences);
}

#[test]
fn enhancer_preserves_stop_words() {
    let enhancer = StochasticEnhancer::new(1.0);
    let input = "The equations and the patterns are in the universe.";
    let result = enhancer.enhance(input);
    assert!(
        result.to_lowercase().contains("the"),
        "Stop word 'the' should be preserved"
    );
}

#[test]
fn enhancer_preserves_trailing_punctuation() {
    let enhancer = StochasticEnhancer::new(1.0);
    let input = "Mathematics enables knowledge.";
    for _ in 0..5 {
        let result = enhancer.enhance(input);
        assert!(
            result.ends_with('.'),
            "Result must end with period, got: {}",
            result
        );
    }
}

#[test]
fn enhancer_preserves_capitalization_on_first_word() {
    let enhancer = StochasticEnhancer::new(1.0);
    let input = "Entropy governs the structure of complexity.";
    for _ in 0..5 {
        let result = enhancer.enhance(input);
        let first_char = result.chars().next().unwrap();
        assert!(
            first_char.is_uppercase(),
            "First char must be uppercase, got: {}",
            result
        );
    }
}

#[test]
fn enhancer_produces_non_empty_output() {
    let enhancer = StochasticEnhancer::new(0.5);
    let input = "The mathematical structure of the universe encodes reality.";
    let result = enhancer.enhance(input);
    assert!(!result.is_empty());
    assert!(result.split_whitespace().count() > 3);
}

#[test]
fn multiple_runs_all_differ() {
    let enhancer = StochasticEnhancer::new(0.7);
    let input = "Mathematics enables the representation of reality. Equations reveal the structure of knowledge. Entropy governs the evolution of complexity.";
    let runs: Vec<String> = (0..5).map(|_| enhancer.enhance(input)).collect();
    let unique: std::collections::HashSet<&String> = runs.iter().collect();
    assert!(unique.len() >= 2, "At least 2 of 5 runs should be distinct");
}

#[test]
fn enhancer_with_curated_word_replaces_known_synonyms() {
    let enhancer = StochasticEnhancer::new(1.0);
    let input = "Mathematics represents truth.";
    let mut found_synonym = false;
    let math_synonyms = ["algebra", "geometry", "calculus", "arithmetic", "analysis"];
    let truth_synonyms = ["reality", "fact", "knowledge", "verity", "actuality"];
    for _ in 0..20 {
        let result = enhancer.enhance(input).to_lowercase();
        if math_synonyms.iter().any(|s| result.contains(s))
            || truth_synonyms.iter().any(|s| result.contains(s))
        {
            found_synonym = true;
            break;
        }
    }
    assert!(
        found_synonym,
        "At least one curated synonym should appear in 20 runs"
    );
}
