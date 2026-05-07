#[derive(Debug, Clone, PartialEq)]
pub enum GenerationMode {
    Sentence,
    Paragraph,
    Essay,
    Summarize,
    Predict,
    Ask,
}

impl GenerationMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Sentence => "Sentence",
            Self::Paragraph => "Paragraph",
            Self::Essay => "Essay",
            Self::Summarize => "Summarize",
            Self::Predict => "Predict",
            Self::Ask => "Ask",
        }
    }

    pub fn icon_class(&self) -> &'static str {
        match self {
            Self::Sentence => "fa-solid fa-pen-nib",
            Self::Paragraph => "fa-solid fa-align-left",
            Self::Essay => "fa-solid fa-book-open",
            Self::Summarize => "fa-solid fa-scissors",
            Self::Predict => "fa-solid fa-wand-magic-sparkles",
            Self::Ask => "fa-solid fa-globe",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Sentence => "Generate a single deterministic sentence from a seed",
            Self::Paragraph => "Generate a multi-sentence paragraph",
            Self::Essay => "Generate a structured essay with title and paragraphs",
            Self::Summarize => "Extract key sentences from a given corpus",
            Self::Predict => "Symbolically continue input text",
            Self::Ask => "Web-search-augmented knowledge synthesis",
        }
    }

    pub fn color_class(&self) -> &'static str {
        match self {
            Self::Sentence => "bg-vect-elevated text-vect-text",
            Self::Paragraph => "bg-vect-elevated text-vect-text",
            Self::Essay => "bg-vect-elevated text-vect-text",
            Self::Summarize => "bg-vect-elevated text-vect-text",
            Self::Predict => "bg-vect-elevated text-vect-text",
            Self::Ask => "bg-vect-accent/20 text-vect-accent",
        }
    }

    pub fn all() -> Vec<GenerationMode> {
        vec![
            GenerationMode::Sentence,
            GenerationMode::Paragraph,
            GenerationMode::Essay,
            GenerationMode::Summarize,
            GenerationMode::Predict,
            GenerationMode::Ask,
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchLink {
    pub title: String,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub id: usize,
    pub role: MessageRole,
    pub content: String,
    pub mode: Option<GenerationMode>,
    pub timestamp: String,
    pub links: Vec<SearchLink>,
}
