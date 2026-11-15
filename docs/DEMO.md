# Three-minute reviewer walkthrough

1. Start the server and open the application. Click **Explore sample workspace**. Three PDFs appear.
2. Ask: **Compare the production RTO and RPO between the 2024 and 2025 incident policies.**
   Expect four hours / 60 minutes for 2024 and one hour / 15 minutes for 2025.
3. Click a citation, inspect its full excerpt, and open the correct PDF page.
4. Ask: **How did the post-incident review deadline change between those policies?**
   Expect five to three business days. Inspect the rewritten question in retrieval details.
5. Ask: **What is the company's maternity leave entitlement?**
   Expect an explicit unavailable answer, without an invented entitlement.
6. Start a new conversation. Confirm it has no prior turns. Reload the page and reopen the previous chat.
7. Select only the 2024 policy and ask about its recovery objectives; evidence must stay within that file.
8. Show the checked-in evaluation JSON and explain both its metrics and its limitations.

Record the actual application and actual model responses. Do not splice simulated answers into a live demo.
Check Groq quota before recording. First search loads embedding/reranker models and is slower than warm searches.
