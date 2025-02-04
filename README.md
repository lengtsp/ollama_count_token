# ollama_count_token

## การใช้งาน

ใช้คำสั่งด้านล่างเพื่อรัน `benchmark.py` พร้อมโหมด verbose และระบุ prompt หลายรายการ:

```bash
python benchmark.py --verbose --prompts "What is the sky blue?" "Write a report on the financials of Nvidia"
```

## ตัวอย่าง result
----------------------------------------------------
        Llama-3.3-70B-Instruct-IQ3_XS.gguf:latest
                Response Rate: 13.34 tokens/s
        
        Stats:
                Response tokens (estimated): 578
                Total time: 43.33s
----------------------------------------------------
        
Average stats:

----------------------------------------------------
        Llama-3.3-70B-Instruct-IQ3_XS.gguf:latest
                Response Rate: 8.64 tokens/s
        
        Stats:
                Response tokens (estimated): 434
                Total time: 50.23s
----------------------------------------------------
