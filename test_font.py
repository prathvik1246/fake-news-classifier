from PIL import ImageFont

try:
    font = ImageFont.truetype("/home/prathvik/fake-news-classifier/Roboto-Regular.ttf", 40)
    print("✅ Font loaded successfully")
except Exception as e:
    print("❌ Font loading failed:", e)

