# Load model directly
#Test the Uploaded Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

local_path = "./legal-summarizer"
# Load the model and tokenizer from the local path
loaded_tokenizer = AutoTokenizer.from_pretrained(local_path)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(local_path)

# Example text to summarize
text_to_summarize = """
EMPLOYMENT AGREEMENT

This Employment Agreement (“Agreement”) is entered into as of May 19, 2025, by and between GlobalTech Solutions LLC, a Delaware limited liability company (“Employer”), having its principal place of business at 1234 Innovation Drive, San Francisco, CA 94105, and John A. Doe (“Employee”), residing at 5678 Maple Street, San Jose, CA 95123.

1. Position and Duties
Employer agrees to employ Employee as a Senior Software Engineer. Employee shall perform all duties and responsibilities customary for such position and as may be assigned by Employer from time to time. Employee agrees to devote full working time, attention, and best efforts to the business of Employer.

2. Term
This Agreement shall commence on June 1, 2025, and shall continue until terminated by either party in accordance with Section 7 of this Agreement.

3. Compensation
Employer shall pay Employee a base salary of $120,000 per annum, payable in accordance with Employer’s standard payroll schedule. Employee may be eligible for bonuses or stock options subject to Employer’s discretion.

4. Benefits
Employee shall be entitled to participate in all employee benefit plans, policies, and programs made available to similarly situated employees, including health insurance, retirement plans, and paid time off.

5. Confidentiality and Non-Disclosure
Employee acknowledges that during employment, they will have access to confidential and proprietary information. Employee agrees not to disclose or use any such information except as required in the course of employment or as authorized by Employer.

6. Non-Competition and Non-Solicitation
For a period of twelve (12) months following termination of employment, Employee shall not engage in any business competitive with Employer nor solicit Employer’s clients or employees.

7. Termination
Either party may terminate this Agreement at any time by providing thirty (30) days written notice. Employer may terminate immediately for cause, including but not limited to breach of this Agreement, misconduct, or violation of law.

8. Governing Law
This Agreement shall be governed by and construed in accordance with the laws of the State of California.

9. Entire Agreement
This Agreement contains the entire understanding between the parties and supersedes all prior agreements or understandings related to the subject matter hereof.

10. Amendments
No amendment or modification shall be effective unless in writing and signed by both parties.

IN WITNESS WHEREOF, the parties have executed this Employment Agreement as of the date first above written.

GlobalTech Solutions LLC
By: Jane M. Smith, CEO

John A. Doe, Employee

"""

# Prepare the text for the model
inputs = loaded_tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
summary_ids = loaded_model.generate(inputs, max_length=300, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode the summary
summary = loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original Text:\n", text_to_summarize)
print("\nGenerated Summary:\n", summary)