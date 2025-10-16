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

# ---------- HIGH-QUALITY FEATURES (replaced) ----------
# These are phrased to be human-understandable when read directly.
BASE_FIELDS = [
    # External covering / morphology
    "has_fur",                      # Has fur or dense hair
    "has_feathers",                 # Has feathers
    "has_scales",                   # Has scales (not feathers/fur)
    "has_shell",                    # Has a hard external shell
    "has_exoskeleton",              # Has an external skeleton (insects, crustaceans)
    "has_wings",                    # Has wings (for flying or gliding)
    "has_fins",                     # Has fins for swimming
    "has_claws_or_paws",            # Has claws, paws, or talons
    "has_horns_or_antlers",         # Has horns, antlers, or similar protrusions
    "has_tusks",                    # Has tusks
    "has_multiple_body_segments",   # Segmented body (e.g., insects, worms)
    "has_tentacles_or_arms",        # Tentacles or arm-like appendages (octopus, squid)

    # Body shape / proportions
    "long_and_slender_body",        # Long, snake-like or elongated body
    "compact_and_stout_body",       # Compact or heavily-built body
    "flattened_body_shape",         # Dorsoventrally flattened (e.g., rays, flounder)
    "very_small_head_relative_to_body", # Head small compared to body

    # Surface patterning / coloration
    "has_spots_or_spotted_pattern", # Spots visible on body
    "has_stripes_or_banded_pattern",# Stripes or banding patterns
    "has_countershading",           # Dark top, light bottom (common in many animals)
    "can_change_color_or_camouflage",# Capable of rapid color change for camouflage/signaling
    "bright_iridescent_coloring",   # Bright or iridescent coloration used for display

    # Locomotion & mobility
    "primarily_walks_or_runs",      # Uses legs to walk/run on land
    "primarily_climbs",             # Regularly climbs trees or vertical surfaces
    "primarily_swims",              # Primarily moves by swimming
    "primarily_flies_or_glides",    # Primarily flies or glides
    "primarily_burrows",            # Lives in burrows or digs to move
    "capable_of_jump_or_leap",      # Capable of powerful jumping or leaping

    # Habitat (phrased as "primarily" or "frequently")
    "primarily_terrestrial",        # Lives mainly on land
    "primarily_aquatic",            # Lives mainly in water
    "primarily_freshwater",         # Mostly found in freshwater (rivers, lakes)
    "primarily_marine",             # Mostly found in saltwater (oceans, seas)
    "frequently_in_forests",        # Often found in forested areas
    "frequently_in_grasslands",     # Often found in grasslands or savanna
    "frequently_in_deserts",        # Often found in arid or desert regions
    "frequently_in_mountains",      # Often found in mountainous areas
    "frequently_in_urban_or_suburban_areas", # Common around human settlements
    "occurs_in_tropical_regions",   # Native to tropical climates
    "occurs_in_temperate_regions",  # Native to temperate climates
    "occurs_in_polar_or_arctic_regions", # Native to polar/arctic climates

    # Activity pattern (human-friendly phrasing)
    "mostly_active_during_day",     # Mostly active during daylight hours
    "mostly_active_at_night",       # Mostly active at night
    "mostly_active_at_dawn_or_dusk",# Most active at sunrise/sunset (crepuscular)

    # Social behavior & grouping
    "lives_in_large_social_groups", # Regularly forms large groups, flocks, colonies
    "lives_in_small_family_groups", # Lives in small family groups or pairs
    "typically_solitary",           # Usually solitary except mating/raising young
    "has_strong_pair_bonding",      # Forms long-term monogamous pairs
    "forms_short_term_associations",# Forms transient groups (seasonal or feeding)

    # Diet & feeding strategy (phrased clearly)
    "eats_meat_regularly",          # Regularly consumes vertebrate or large animal prey
    "eats_insects_or_small_invertebrates", # Eats insects or small invertebrates often
    "eats_plants_regularly",        # Regularly consumes plant material
    "eats_fish_or_aquatic_prey",    # Diet primarily includes fish or aquatic animals
    "is_opportunistic_omnivore",    # Eats both plants and animals opportunistically
    "scavenges_dead_animal_matter", # Frequently scavenges carrion or leftovers
    "specialized_pollen_or_nectar_feeder", # Feeds on nectar/pollen (e.g., many birds/insects)

    # Predation / defense
    "is_apex_or_top_predator",      # Top predator in its environment
    "uses_ambush_or_camouflage_to_hunt", # Hunts using stealth/ambush/camouflage
    "has_poison_or_venom",          # Produces venom or poison delivered by bite/sting
    "uses_strong_physical_defense", # Uses armor, spines, shells for defense
    "can_release_noxious_chemicals",# Releases foul chemicals for defense
    "can_detach_body_parts_to_escape", # Can autotomize (drop tail/limb) to escape predators

    # Reproduction & lifecycle
    "lays_eggs",                    # Reproduces by laying eggs
    "gives_live_birth",             # Gives live birth to young
    "provides_parental_care",       # Provides extended parental care to offspring
    "many_small_offspring_at_once", # Produces many small offspring (r-strategy)
    "few_large_offspring_with_investment", # Few offspring with high parental investment

    # Sensory systems & communication
    "has_keen_vision",              # Vision is a primary sense (good eyesight)
    "has_keen_hearing",             # Hearing is a primary sense (good ears)
    "uses_chemical_sensing_or_olfaction", # Relies heavily on smell/chemical cues
    "uses_echolocation_or_sonar",   # Uses echolocation (bats, some marine mammals)
    "communicates_with_vocal_sounds",# Commonly communicates via vocalizations
    "communicates_with_visual_displays",# Uses body display or color for communication

    # Physiology & environmental tolerances
    "tolerates_cold_conditions",    # Adapted to cold climates
    "tolerates_heat_and_dry_conditions", # Adapted to hot/dry climates
    "can_withstand_low_oxygen_in_water", # Survives in low-oxygen aquatic conditions
    "can_survive_long_periods_without_food", # Can fast for long periods

    # Interaction with humans & conservation
    "commonly_kept_as_pet",         # Commonly kept as a pet
    "commonly_domesticated_for_farm_use", # Domesticated / farmed species
    "often_seen_near_humans",       # Frequently observed near humans or human structures
    "has_significant_economic_value",# Harvested/valued for food, materials, or trade
    "is_threatened_or_endangered",  # Conservation status threatened/endangered
    "is_important_in_cultural_symbolism", # Important symbolically/culturally to people

    # Misc utilities / niche behaviors
    "migrates_seasonally",          # Undertakes seasonal migration
    "builds_complex_structures",    # Builds nests, burrows, hives, or other structures
    "engages_in_tool_use",          # Known to use tools in the wild
    "digests_wood_or_cellulose",    # Digestive specialization for wood/cellulose (termites)
    "lives_on_or_attaches_to_other_animals", # Parasitic or commensal lifestyle
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

    # Load input animals
    try:
        input_df = pd.read_csv(input_csv)
        all_animals = input_df['animal_name'].dropna().unique().tolist()
        print(f"📋 Found {len(all_animals)} animals in {input_csv}")
    except FileNotFoundError:
        print(f"❌ Missing {input_csv}. Please create it first.")
        return
    except Exception as e:
        print(f"❌ Error reading {input_csv}: {e}")
        return

    # Load already processed animals
    processed = set()
    write_header = True
    if os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv)
            processed = set(existing['animal_name'])
            write_header = False
            print(f"♻️  Resuming: {len(processed)} animals already processed")
        except Exception as e:
            print(f"⚠️  Could not read existing {output_csv}: {e}")

    animals_to_do = [a for a in all_animals if a not in processed]
    
    if not animals_to_do:
        print("✅ All animals already processed!")
        return
    
    print(f"🚀 Processing {len(animals_to_do)} animals with {MAX_WORKERS} workers...")

    # Process animals in parallel with batch writing
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

    # Write remaining batch
    if batch_buffer:
        write_batch_to_csv(batch_buffer, output_csv, write_header)

    # Summary
    print(f"\n{'='*50}")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output saved to: '{output_csv}'")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
