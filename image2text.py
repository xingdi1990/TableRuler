import torch
from transformers import LlamaTokenizer, AutoProcessor, AutoModelForCausalLM, OwlViTProcessor, OwlViTForObjectDetection,InstructBlipProcessor, InstructBlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
from utils import *
import argparse
import json
import asyncio
import base64
import mimetypes
from typing import List, Dict, Any
import os
from openai import AsyncOpenAI
import time

def build_finetuning_prompt(base_text: str) -> str:
    return (
        f"Below is the image of table, as well as some html content that was previously extracted for it. "
        f"Just return the refined html content of this table.\n"
        f"Do not hallucinate.\n"
        f"HTML_START\n{base_text}\nHTML_END"
    )

def get_all_jsons(directory):
    """Get all file paths under a directory and return them as a list."""
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_paths.append(os.path.join(root, file))

    return file_paths

def load_tables_json(json_file):
    """Load and process tables from JSON file one by one.
    
    Args:
        json_file (str): Path to the JSON file containing table data
        
    Yields:
        dict: Each table entry from the JSON file
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            tables = json.load(f)
            for table in tables:
                yield table
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

def process_table(table):
    """Process a single table entry.
    
    Args:
        table (dict): Table entry containing page, img_path, types, and sentence
        
    Returns:
        dict: Processed table data
    """
    # Example processing - you can modify this based on your needs
    processed = {
        'page': table.get('page'),
        'img_path': table.get('img_path'),
        'types': table.get('types'),
        'sentence': table.get('sentence')
    }
    return processed


# device = "cuda" if torch.cuda.is_available() else "cpu"
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# cropper = AllCrop(size=(224,224), stride=(256, 210)) #ori size:480x856 (68, 104) (128, 158) (256, 210)


def cogvlm(model, image_paths, mode = 'chat', root_path = None, model_path = 'lmsys/vicuna-7b-v1.5'):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    query= 'Below is the image of one table, just return the plain text representation of this table image as if you were reading it naturally.'
    # if root_path != None:
    #     image_paths = sorted(get_all_paths(root_path))

    batch_images = [Image.open(p) for p in image_paths]
    # image = Image.open(image_path)
    description = []

    # for count, query in enumerate(queries):
    for image in batch_images:
        if mode == 'chat':
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # vqa mode
        else:
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image],
                                                        template_version='vqa')  # vqa mode

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            description.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return description


async def process_single_table(client: AsyncOpenAI, table: Dict, data_root_name: str, system_prompt: str) -> Dict:
    """Process a single table asynchronously."""
    try:
        processed_table = process_table(table)
        encoded_string = base64.b64encode(open(processed_table['img_path'], 'rb').read()).decode('utf-8')
        anchor_text = processed_table['sentence']
        mime_type, _ = mimetypes.guess_type(processed_table['img_path'])
        data_uri = f"data:{mime_type};base64,{encoded_string}"
        
        messages = [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""HTML_START\n{anchor_text}\nHTML_END"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ]

        # Send request to OpenAI
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
            max_tokens=1024
        )

        refined_html = response.choices[0].message.content
        
        # Create output dictionary
        output = {
            'page': processed_table['page'],
            'img_path': processed_table['img_path'],
            'types': processed_table['types'],
            'sentence': processed_table['sentence'],
            'refined_sentence': refined_html
        }

        # Save individual result with proper encoding
        json_path = processed_table['img_path'].split('/')[-1].split('.')[0]
        output_path = f'{data_root_name}/table_jsons/{json_path}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        return output
    except Exception as e:
        print(f"Error processing table {processed_table['img_path']}: {e}")
        return None

def get_description_json(data_root_name: str):
    """Process all tables from all JSON files concurrently."""
    # Initialize async client
    client = AsyncOpenAI(
        base_url="https://api.netmind.ai/inference-api/openai/v1",
        api_key="098ba1186ed849bca1180a75383075b7",
    )
    
    all_json_paths = get_all_jsons(data_root_name)
    print(f"Found {len(all_json_paths)} JSON files")
    
    system_prompt = """
    Below is the image of table, as well as some html content that was previously extracted for it. Just return the refined html content of this table. Do not hallucinate.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(data_root_name, 'table_images'), exist_ok=True)
    
    # Concatenate all tables from all JSON files
    all_tables = []
    for json_path in tqdm(all_json_paths, desc="Loading JSON files"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Store the source file name with each table for organization
                for table in data:
                    table['source_file'] = json_path.split('/')[-1].split('.')[0]
                all_tables.extend(data)
        except Exception as e:
            print(f"Error loading file {json_path}: {e}")
            continue
    
    print(f"\nTotal tables to process: {len(all_tables)}")
    
    async def process_all_tables():
        results = []
        
        with tqdm(total=len(all_tables), desc="Processing tables") as pbar:
            for table in all_tables:
                # Process single table
                result = await process_single_table(client, table, data_root_name, system_prompt)
                results.append(result)
                pbar.update(1)
                
        return results
    
    # Run all tasks
    all_results = asyncio.run(process_all_tables())
    
    # Organize results by source file
    results = {}
    for result in all_results:
        if result is not None:
            source_file = result.pop('source_file', 'unknown')  # Remove and get source_file
            if source_file not in results:
                results[source_file] = []
            results[source_file].append(result)
    
    return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/xing/TableRuler/datasets/orbit_v1/azure_blocks')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    data_root_name = args.data
    
    try:
        start_time = time.time()
        results = get_description_json(data_root_name)
        end_time = time.time()
        
        # Print summary statistics
        print("\nProcessing Summary:")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Files processed: {len(results)}")
        total_tables = sum(len(tables) for tables in results.values())
        print(f"Tables processed: {total_tables}")
        print(f"Average time per table: {(end_time - start_time) / total_tables:.2f} seconds")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()



