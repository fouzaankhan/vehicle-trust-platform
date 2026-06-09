import pandas as pd
import random
import hashlib

random.seed(42)

LEGIT_TEMPLATES = [
    "Well maintained {year} {make} {model}. Single owner, all service records available. "
    "Minor wear on seats. New tires fitted last month. Selling due to upgrade.",

    "Used {year} {make} {model} for sale. {km}k miles on odometer. "
    "No accidents, clean title. AC works great. Available for inspection anytime.",

    "Selling my {year} {make} {model}. Bought new, used daily for commute. "
    "Regular oil changes done at dealership. Small scratch on rear bumper, otherwise clean.",

    "{year} {make} {model} in good condition. Second owner. "
    "All four tires replaced recently. Engine runs smooth. Test drive welcome.",

    "Family car, {year} {make} {model}. Used for school runs and weekend trips. "
    "Full service history. Nonsmoker vehicle. Priced to sell quickly."
]

SCAM_TEMPLATES = [
    "URGENT SALE - {year} {make} {model}. I am currently posted abroad with the army "
    "and need to sell immediately. God fearing seller. Will ship via escrow. "
    "100% genuine deal. Only serious buyers. Price is negotiable, contact on WhatsApp.",

    "Selling {year} {make} {model} BELOW MARKET due to emergency. Shifting abroad next week. "
    "No time to negotiate. Trust me this is genuine. Send advance to confirm interest. "
    "WhatsApp only: will share photos privately.",

    "Best deal on {year} {make} {model}. I am honest god fearing person. "
    "Car is in excellent condition guaranteed. Accident free 100%. "
    "Urgent sale due to medical emergency. Call now limited time offer!!!",

    "{year} {make} {model} for sale. Owner moving overseas urgently. "
    "Price is firm, serious buyers only. No scammers please. "
    "Payment via bank transfer only, car will be delivered after payment confirmed.",

    "ASAP SALE {year} {make} {model}. Below market value, today only. "
    "God is my witness car is perfect. Army deployment forces this sale. "
    "Escrow payment preferred. WhatsApp me for more details and private photos."
]

MAKES = ["Ford", "Toyota", "Honda", "BMW", "Chevrolet", "Hyundai", "Volkswagen"]
MODELS = ["Camry", "Civic", "F-150", "3 Series", "Silverado", "Elantra", "Golf"]

rows = []
for i in range(1000):
    make = random.choice(MAKES)
    model = random.choice(MODELS)
    year = random.randint(2005, 2022)
    km = random.randint(10, 250)
    is_scam = random.random() < 0.25  # 25% scam rate

    template = random.choice(SCAM_TEMPLATES if is_scam else LEGIT_TEMPLATES)
    description = template.format(year=year, make=make, model=model, km=km)

    rows.append({
        "listing_id": hashlib.md5(f"{i}{description}".encode()).hexdigest()[:10],
        "make": make,
        "model": model,
        "year": year,
        "description": description,
        "is_scam": int(is_scam)
    })

df = pd.DataFrame(rows)
df.to_csv("data/processed/listing_descriptions.csv", index=False)
print(f"Generated {len(df)} descriptions ({df['is_scam'].sum()} scam, {(~df['is_scam'].astype(bool)).sum()} legit)")