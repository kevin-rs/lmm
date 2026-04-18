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

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Sentence => "✍️",
            Self::Paragraph => "📄",
            Self::Essay => "📖",
            Self::Summarize => "✂️",
            Self::Predict => "🔮",
            Self::Ask => "🌐",
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
            Self::Sentence => "bg-vect-violet text-white",
            Self::Paragraph => "bg-violet-700 text-white",
            Self::Essay => "bg-violet-600 text-white",
            Self::Summarize => "bg-vect-cyan text-vect-bg",
            Self::Predict => "bg-indigo-600 text-white",
            Self::Ask => "bg-teal-600 text-white",
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
