use tokio_util::sync::CancellationToken;

pub enum ShutdownSignal {
    CtrlC,
    Custom(CancellationToken),
    None,
}

impl Default for ShutdownSignal {
    fn default() -> Self { Self::CtrlC }
}

impl ShutdownSignal {
    pub fn into_token(self) -> Option<CancellationToken> { 
        match self { 
            ShutdownSignal::CtrlC => Some(CancellationToken::new()), 
            ShutdownSignal::Custom(token) => Some(token), 
            ShutdownSignal::None => None 
        } 
    }
}    