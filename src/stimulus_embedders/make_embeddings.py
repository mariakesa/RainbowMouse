from dotenv import load_dotenv
load_dotenv()
import os
from transformers import ViTImageProcessor, ViTModel
from stim_emb import MakeEmbeddings

object_recognition_vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
object_recognition_vit_model= ViTModel.from_pretrained("google/vit-base-patch16-224")

clip_processor = ViTImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = ViTModel.from_pretrained("openai/clip-vit-base-patch16")

dino_processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
dino_model = ViTModel.from_pretrained("facebook/dino-vitb16")

models={
    'object_recognition_vit_model': (object_recognition_vit_model, object_recognition_vit_processor),
    'clip_model': (clip_model, clip_processor),
    'dino_model': (dino_model, dino_processor)
}


for m in models.keys():
    model, processor = models[m]
    # Save the model and processor to the cache
    embedder=MakeEmbeddings(processor, model)
    embedder.execute()
    print(f"Embeddings for {m} saved successfully.")
    
    
