import os
import time
import json
import re
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# --- Configuration ---
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("OPENROUTER_API_KEY")

VALID_CLASSES = ["Mammal", "Bird", "Reptile", "Fish", "Amphibian", "Insect", "Arachnid", "Crustacean", "Mollusc"]
VALID_SIZES = ["Tiny", "Small", "Medium", "Large", "Very Large", "Massive"]

# ---------- HIGH-QUALITY DISCRIMINATIVE FEATURES ----------
# Organized from fundamental biology to specific behaviors for clarity.
BASE_FIELDS = [
    # --- Fundamental Biology & Physiology ---
    "is_vertebrate",                  # Has a backbone/internal skeleton (mammals, birds, fish, etc.)
    "is_invertebrate",                # Lacks a backbone (insects, molluscs, etc.)
    "is_warm_blooded",                # Endothermic: maintains constant internal body temperature
    "is_cold_blooded",                # Ectothermic: body temperature relies on the environment
    "has_gills_for_breathing",        # Breathes water using gills (at least in some life stages)
    "has_lungs_for_breathing",        # Breathes air using lungs
    "hibernates_or_enters_torpor",    # Undergoes a state of inactivity for winter/cold periods
    "is_bioluminescent",              # Can produce its own light (e.g., firefly, anglerfish)

    # --- External Covering & Morphology ---
    "has_fur_or_hair",                # Covered in fur or hair
    "has_feathers",                   # Has feathers
    "has_scales",                     # Has scales (e.g., fish, reptiles)
    "has_leathery_or_slimy_skin",     # Skin is leathery, smooth, or slimy (e.g., amphibians, worms)
    "has_hard_exoskeleton",           # Has a chitinous external skeleton (e.g., insects, crustaceans)
    "has_shell",                      # Has a hard external shell (e.g., turtle, snail)
    "has_wings",                      # Has wings (for flying or gliding)
    "has_fins",                       # Has fins for swimming
    "has_claws_paws_or_talons",       # Has claws, paws, or talons
    "has_hooves",                     # Has hooves
    "has_horns_or_antlers",           # Has horns, antlers, or similar protrusions
    "has_distinct_beak_or_bill",      # Has a beak/bill instead of a mouth with teeth (birds, turtles)
    "has_tusks",                      # Has tusks
    "has_tentacles_or_arms",          # Tentacles or arm-like appendages (octopus, squid)
    "has_prehensile_tail",            # Tail is capable of grasping

    # --- Anatomy & Body Plan ---
    "has_no_legs",                    # Legless
    "has_two_legs",                   # Bipedal
    "has_four_legs",                  # Quadrupedal
    "has_six_legs",                   # A key identifier for insects
    "has_eight_legs",                 # A key identifier for arachnids
    "has_more_than_eight_legs",       # For myriapods, crustaceans etc.
    "has_compound_eyes",              # Eyes made of many individual units (common in arthropods)
    "has_segmented_body",             # Body is clearly divided into segments (e.g., insects, worms)
    "long_and_slender_body",          # Long, snake-like or elongated body
    "compact_and_stout_body",         # Compact, rounded, or heavily-built body
    "radially_symmetrical_body",      # Body parts arranged around a central axis (e.g., starfish, jellyfish)

    # --- Locomotion ---
    "primarily_walks_or_runs",        # Uses legs to walk/run on land
    "primarily_climbs",               # Regularly climbs trees, rocks, or other vertical surfaces
    "primarily_swims",                # Primarily moves by swimming
    "primarily_flies_or_glides",      # Primarily moves by flying or gliding
    "primarily_burrows_or_digs",      # Lives in burrows or digs as a primary mode of movement
    "is_sessile_as_adult",            # Is immobile and fixed in one place as an adult (e.g., barnacles, corals)
    "moves_by_jumping_or_leaping",    # Primary mode of locomotion is jumping/leaping

    # --- Habitat ---
    "is_terrestrial",                 # Lives exclusively on land
    "is_aquatic",                     # Lives exclusively in water
    "is_semi_aquatic",                # Lives both on land and in water (e.g., frog, beaver)
    "is_arboreal",                    # Lives primarily in trees
    "is_fossorial",                   # Adapted to digging and lives underground
    "lives_in_freshwater",            # Mostly found in freshwater (rivers, lakes)
    "lives_in_marine_water",          # Mostly found in saltwater (oceans, seas)
    "lives_in_polar_regions",         # Native to polar/arctic climates
    "lives_in_forests_or_woodlands",  # Often found in forested areas
    "lives_in_grasslands_or_savannas",# Often found in grasslands or savanna
    "lives_in_deserts_or_arid_regions",# Often found in arid or desert regions
    "common_in_urban_areas",          # Thrives in or near human cities and suburbs

    # --- Diet & Feeding Strategy ---
    "is_carnivore",                   # Primarily eats meat
    "is_herbivore",                   # Primarily eats plants
    "is_omnivore",                    # Eats both plants and animals
    "is_insectivore",                 # Primarily eats insects
    "is_piscivore",                   # Primarily eats fish
    "is_filter_feeder",               # Feeds by filtering particles from water
    "is_scavenger",                   # Frequently scavenges carrion
    "eats_nectar_pollen_or_fruit",    # Feeds on nectar, pollen, or fruit
    "is_parasitic",                   # Lives on or in a host organism, causing it harm

    # --- Reproduction & Lifecycle ---
    "lays_eggs",                      # Reproduces by laying eggs (Oviparous)
    "gives_live_birth",               # Gives live birth to young (Viviparous)
    "undergoes_metamorphosis",        # Has distinct larval and adult stages (e.g., butterflies, frogs)
    "provides_extensive_parental_care",# Parents actively raise, feed, and protect young
    "has_no_parental_care",           # Offspring are independent from birth/hatching
    "carries_young_in_pouch",         # Marsupial reproduction (e.g., kangaroo)

    # --- Social Behavior ---
    "is_highly_social_or_eusocial",   # Lives in large, complex, cooperative societies (e.g., ants, bees, meerkats)
    "lives_in_small_groups_or_pairs", # Forms small family units, packs, or monogamous pairs
    "is_solitary",                    # Usually lives alone except to mate
    "is_territorial",                 # Actively defends a specific area

    # --- Defense & Predation ---
    "is_apex_predator",               # Top predator in its environment with no natural predators
    "uses_camouflage_or_mimicry",     # Blends in with the environment or mimics other animals
    "has_venom_or_poison",            # Produces venom (injected) or poison (ingested/touched)
    "uses_armor_or_spines_for_defense", # Uses physical armor, shells, or spines for defense
    "releases_noxious_chemicals",     # Sprays or secretes foul chemicals for defense (e.g., skunk)
    "relies_on_speed_to_escape",      # Primary defense is to flee quickly

    # --- Human Interaction & Special Traits ---
    "is_domesticated",                # Has been selectively bred by humans
    "migrates_seasonally",            # Undertakes long-distance seasonal migration
    "builds_complex_structures",      # Builds nests, dams, hives, webs, or other structures
    "uses_tools",                     # Known to use objects as tools
    "communicates_with_complex_vocalizations", # Uses a wide range of sounds/songs for communication
    "is_nocturnal",                   # Primarily active at night
    "is_diurnal",                     # Primarily active during the day
]
# -----------------------------------------------------

CLASS_FIELDS = [f"is_{c.lower()}" for c in VALID_CLASSES]
SIZE_FIELDS = [f"is_{s.lower().replace(' ', '_')}" for s in VALID_SIZES]
FIELDNAMES = ["animal_name"] + CLASS_FIELDS + SIZE_FIELDS + BASE_FIELDS

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o"
YOUR_SITE_URL = "http://localhost"
APP_NAME = "Animal Predictor DataGen"
MAX_RETRIES = 3
RETRY_DELAY = 0.5
BATCH_WRITE_SIZE = 20
MAX_WORKERS = 20

# Thread-safe file writing
write_lock = threading.Lock()

def extract_json_from_string(text: str) -> str | None:
    """Extract JSON object from text, handling markdown code blocks."""
    # Try to extract from markdown code block first
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block:
        return code_block.group(1)
    # Fall back to finding first JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def get_animal_characteristics(animal_name: str) -> dict | None:
    """Fetch characteristics for a single animal with retries."""
    json_schema_for_prompt = "".join(
        f'- "{field}": (Number) A rating from 0.0 to 1.0.\n'
        for field in CLASS_FIELDS + SIZE_FIELDS + BASE_FIELDS
    )

    prompt = f"""You are an expert zoologist creating a fuzzy-logic dataset for a machine learning model.
Provide JSON characteristics for: "{animal_name}"

Rules:
- Respond ONLY with a single minified JSON object, no other text
- Each key in schema below must be present, numeric [0.0–1.0]
- Use 0.0 for definitely false/absent, 1.0 for definitely true/present
- Use intermediate values for partial matches or uncertainty

Required fields:
{json_schema_for_prompt}

Respond with ONLY the JSON object."""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": APP_NAME,
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3  # Lower temperature for more consistent outputs
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            raw = response.json()['choices'][0]['message']['content']
            json_str = extract_json_from_string(raw)
            
            if not json_str:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"⚠️  No JSON found for {animal_name}")
                    return None
            
            d = json.loads(json_str)
            d["animal_name"] = animal_name
            
            # Validate all required fields are present
            missing = [f for f in FIELDNAMES[1:] if f not in d]
            if missing:
                print(f"⚠️  Missing fields for {animal_name}: {missing[:3]}...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            
            return d
            
        except requests.exceptions.Timeout:
            print(f"⏱️  Timeout for {animal_name} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error for {animal_name}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Parse error for {animal_name}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    return None

def write_batch_to_csv(batch_data: list, output_csv: str, write_header: bool):
    """Thread-safe batch writing to CSV."""
    if not batch_data:
        return
    
    with write_lock:
        df_batch = pd.DataFrame(batch_data)[FIELDNAMES]
        mode = 'a' if os.path.exists(output_csv) and not write_header else 'w'
        df_batch.to_csv(output_csv, mode=mode, header=write_header, index=False)

def main():
    print("--- ⚡ Fast Animal Data Generator ---")
    input_csv = "animals.csv"
    output_csv = "animalfulldata.csv"

    # --- Step 1: Clean the input CSV ---
    print("\n--- 🧼 Cleaning Input File ---")
    try:
        input_df = pd.read_csv(input_csv)
        original_count = len(input_df)
        
        # Standardize: trim whitespace and convert to title case to catch more duplicates
        input_df['animal_name'] = input_df['animal_name'].str.strip().str.title()
        
        # Drop empty rows and duplicates
        input_df.dropna(subset=['animal_name'], inplace=True)
        input_df.drop_duplicates(subset=['animal_name'], keep='first', inplace=True)
        cleaned_count = len(input_df)
        
        duplicates_removed = original_count - cleaned_count
        
        if duplicates_removed > 0:
            print(f"🧹 Found and removed {duplicates_removed} duplicate/empty row(s).")
            input_df.to_csv(input_csv, index=False)
            print(f"💾 Cleaned '{input_csv}' has been saved.")
        else:
            print("✅ Input file is already clean. No duplicates found.")

    except FileNotFoundError:
        print(f"❌ Error: Missing '{input_csv}'. Please create it with an 'animal_name' column.")
        return
    except Exception as e:
        print(f"❌ Error cleaning '{input_csv}': {e}")
        return

    # --- Step 2: Ask user about regeneration ---
    print("\n--- ⚙️ Generation Mode ---")
    regenerate_all = input("🔄 Do you want to regenerate all data from scratch? (y/n): ").strip().lower()

    if regenerate_all == 'y':
        if os.path.exists(output_csv):
            try:
                os.remove(output_csv)
                print(f"🗑️  Removed existing '{output_csv}' for a full regeneration.")
            except OSError as e:
                print(f"❌ Could not remove '{output_csv}': {e}. Please remove it manually.")
                return
        else:
            print("✨ Starting a fresh generation.")
    else:
        print(" Gearing up to process only new animals.")


    # --- Step 3: Load animals and determine what to process ---
    all_animals = input_df['animal_name'].tolist()
    
    processed = set()
    write_header = True
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'animal_name' in existing_df.columns:
                 processed = set(existing_df['animal_name'])
            write_header = False
            print(f"♻️  Resuming: {len(processed)} animals already processed.")
        except pd.errors.EmptyDataError:
            print(f"⚠️  '{output_csv}' is empty. Starting with a new header.")
        except Exception as e:
            print(f"⚠️  Could not read existing '{output_csv}': {e}")

    animals_to_do = [a for a in all_animals if a not in processed]
    
    if not animals_to_do:
        print("\n✅ All animals already processed!")
        return
    
    # --- Step 4: Process data in parallel ---
    print(f"\n🚀 Processing {len(animals_to_do)} new animals with {MAX_WORKERS} workers...")

    batch_buffer = []
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_animal_characteristics, a): a for a in animals_to_do}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="🐾 Fetching"):
            animal = futures[future]
            try:
                data = future.result()
                if data:
                    batch_buffer.append(data)
                    successful += 1
                    
                    # Write batch when buffer is full
                    if len(batch_buffer) >= BATCH_WRITE_SIZE:
                        write_batch_to_csv(batch_buffer, output_csv, write_header)
                        write_header = False
                        batch_buffer = []
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Unexpected error for {animal}: {e}")
                failed += 1

    # Write any remaining data in the buffer
    if batch_buffer:
        write_batch_to_csv(batch_buffer, output_csv, write_header)

    # --- Step 5: Final Summary ---
    print(f"\n{'='*50}")
    print("🏁 Generation Complete!")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output saved to: '{output_csv}'")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()