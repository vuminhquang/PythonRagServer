from typing import Dict, Optional, List
import spacy
from datetime import datetime
from hashlib import md5
from langdetect import detect
import asyncio
from concurrent.futures import ThreadPoolExecutor

class MultilingualMetadataEnricher:
    SUPPORTED_MODELS = {
        'en': 'en_core_web_sm',
        'vi': 'vi_core_news_lg',  # Vietnamese
        'zh': 'zh_core_web_sm',   # Chinese
        'ja': 'ja_core_news_sm',  # Japanese
        'ko': 'ko_core_news_sm',  # Korean
    }
    
    def __init__(self):
        self.nlp_models = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load_model(self, lang_code: str):
        """Lazy load language models when needed"""
        if lang_code not in self.nlp_models and lang_code in self.SUPPORTED_MODELS:
            try:
                # Run model loading in thread pool to avoid blocking
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    spacy.load,
                    self.SUPPORTED_MODELS[lang_code]
                )
                self.nlp_models[lang_code] = model
            except OSError:
                print(f"Model {self.SUPPORTED_MODELS[lang_code]} not found. Installing...")
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    spacy.cli.download,
                    self.SUPPORTED_MODELS[lang_code]
                )
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    spacy.load,
                    self.SUPPORTED_MODELS[lang_code]
                )
                self.nlp_models[lang_code] = model

    async def process_text(self, content: str, lang_code: str):
        """Process text with appropriate language model"""
        await self.load_model(lang_code)
        if lang_code in self.nlp_models:
            return self.nlp_models[lang_code](content)
        return None

    async def extract_language_specific_features(self, doc, lang_code: str) -> Dict:
        """Extract language-specific features based on the language"""
        features = {}
        
        if lang_code == 'vi':
            # Vietnamese-specific features
            features.update({
                'syllable_count': len(doc.text.split()),
                'vietnamese_particles': [
                    token.text for token in doc 
                    if token.pos_ in ['PART']
                ]
            })
            
        elif lang_code in ['zh', 'ja']:
            # Chinese/Japanese-specific features
            features.update({
                'character_count': len(doc.text),
                'chinese_particles': [
                    token.text for token in doc 
                    if token.pos_ == 'PART'
                ]
            })
            
        return features

    async def enrich(self, content: str, original_metadata: Optional[Dict] = None) -> Dict:
        # Detect language
        lang_code = detect(content)
        
        # Process with appropriate model
        doc = await self.process_text(content, lang_code)
        
        # Base metadata (language-independent)
        metadata = {
            "document_id": md5(content.encode()).hexdigest(),
            "timestamp": datetime.utcnow().isoformat(),
            "char_count": len(content),
            "language": lang_code,
            "language_confidence": 1.0,  # You might want to add actual confidence scoring
        }
        
        if doc is not None:
            # Add language-specific NLP features
            metadata.update({
                "entities": [
                    {
                        "text": ent.text,
                        "label": ent.label_
                    } for ent in doc.ents
                ],
                "keywords": [
                    token.text for token in doc 
                    if not token.is_stop and not token.is_punct
                ],
                "sentence_count": len(list(doc.sents)),
                
                # Content type indicators
                "has_numbers": any(token.like_num for token in doc),
                "has_urls": any(token.like_url for token in doc),
                "has_emails": any("@" in token.text for token in doc),
            })
            
            # Add language-specific features
            lang_features = await self.extract_language_specific_features(doc, lang_code)
            metadata.update(lang_features)
        
        # Merge with original metadata
        if original_metadata:
            metadata.update(original_metadata)
            
        return metadata