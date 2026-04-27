import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "ilsp/Llama-Krikri-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to('cpu')

def correct_transcript_krikri(raw_text):
    system_prompt = (
        "Ο χρήστης θα σου δώσει ένα κείμενο που προέρχεται από αυτόματη αναγνώριση ομιλίας (ASR). "
        "Ενέργησε ως αυστηρός διορθωτής (proofreader)."
        "Διόρθωσε μόνο τα ορθογραφικά και γραμματικά λάθη. ΜΗΝ αντικαθιστάς λέξεις με συνώνυμα. "
        "ΜΗΝ αλλάζεις το ύφος ή το λεξιλόγιο. Αν μια λέξη είναι σωστή, κράτησέ την όπως είναι."
        "ΜΗΝ προσθέσεις σχόλια, εισαγωγές ή επίλογο."
        "Επίστρεψε ΜΟΝΟ το διορθωμένο κείμενο."
    )

    user_message = f"Διόρθωσε το παρακάτω κείμενο:\n\n{raw_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=len(raw_text) + 100,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

    return response.strip()

if __name__ == "__main__":
    bad_transcript = "που χέρουνταν ανόχλητα τα μυρωδατά αγριολούλουβα"

    corrected = correct_transcript_krikri(bad_transcript)

    print("-" * 30)
    print(f"Original:  {bad_transcript}")
    print(f"Corrected: {corrected}")
    print("-" * 30)