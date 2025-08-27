import requests
from bs4 import BeautifulSoup
import json
import re

def scrape_antai_med_info(med_name: str, dosage: str):
    """
    根據藥品名稱和劑量，爬取安泰醫院網站的藥物資訊。
    """
    search_keyword = med_name.split(" ")[0]
    
    base_url = "https://www.antai.tw/medicine_list.asp"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }
    params = {
        'tkeyword': search_keyword
    }

    try:
        print(f"步驟 1: 正在向安泰醫院網站查詢藥名關鍵字 '{search_keyword}'...")
        response = requests.get(base_url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'

        print("步驟 2: 成功收到回應，正在解析所有結果...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        result_rows = soup.find_all('tr', class_='style_10')

        if not result_rows:
            print(f"解析錯誤：用關鍵字 '{search_keyword}' 找不到任何藥物資料列。")
            return None
        
        print(f"找到 {len(result_rows)} 筆相關結果，正在比對最佳項目...")

        best_match_row = None
        # 我們直接選取第一筆最相關的結果
        best_match_row = result_rows[0]
        
        # 從最佳匹配的列中提取資料
        columns = best_match_row.find_all('td')
        
        img_tag = columns[0].find('img')
        image_url = "https://www.antai.tw/" + img_tag['src'] if img_tag and 'src' in img_tag.attrs else "https://via.placeholder.com/100x100.png?text=No+Image"
        name = columns[1].text.strip()
        side_effects = columns[4].text.strip()

        print("步驟 3: 成功解析所有資料！")

        return {
            "name": name,
            "image_url": image_url,
            "side_effects": side_effects if side_effects else "查無此藥品的副作用資訊。"
        }

    except requests.exceptions.RequestException as e:
        print(f"爬取安泰醫院網站時發生網路錯誤: {e}")
        return None
    except Exception as e:
        print(f"發生未知錯誤: {e}")
        return None

if __name__ == "__main__":
    # ✨✨ 我們改用確定存在的中文名稱「複方」來測試 ✨✨
    test_name = "複方" 
    test_dosage = "" 
    
    print(f"--- 正在測試爬取藥品: {test_name} ---")
    
    result = scrape_antai_med_info(test_name, test_dosage)
    
    print("\n--- 爬取完成 ---")
    
    if result:
        print("✅ 成功找到資料！\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("❌ 找不到資料或爬取失敗。")