require('dotenv').config(); 
const axios = require('axios');

async function classifyText(rawdata, link) {
    const prompt = `
  다음은 게시물 내용입니다:

  """${rawdata}"""

게시물을 아래 기준에 따라 분석하여 정확하게 JSON으로 출력해주세요.

---

📌 [분석 기준]

1. 카테고리는 아래 목록 중 최대 3개까지 선택하세요:  
[교육, 공모, 경제, 문화, 미디어, 건강, 환경, 창업, 음식, 과학, 행사, 뷰티, 쇼핑, 인턴, 대회, 카페, 여행, 마케팅]  
※ 아래 카테고리 목록 외의 단어는 절대 사용하지 마세요. 예: "운동", "체육", "채용" 등은 허용되지 않습니다.
- 카테고리는 항상 배열로 반환하세요.
-이모티콘은 제거하세요.

2. 각 항목의 출력 규칙:
- 'title': 게시물 제목
- 'description': 대상자, 자격, 요약 등 중요 정보 한 줄 요약
- 'start_date': 시작일 (YYYY-MM-DD) — 없다면 오늘 날짜로 대체
- 'end_date': 종료일 (YYYY-MM-DD) — '상시모집' 같은 표현은 "9999-12-31"로 고정  
- 'image': 이미지 URL 또는 null
- 'link': 그대로 유지 ("""${link}""")

예시:

{
  "title": "제목",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "description": "요약 내용",
  "categories": "[카테고리]",
  "image": "https://example.com/image.jpg",
  "link": "https://example.com/post"
}

`;

    const response = await axios.post(

      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-3.5-turbo',  // GPT-3.5 모델 (텍스트 기반)
        messages: [
            { role: 'user', content: prompt}
        ],
        temperature: 0.3               // 응답 끝을 정의할 토큰 (선택 사항)
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,  // API 키 헤더에 포함
        },
      }
    );

    const validCategories = [ 
      "교육", "공모", "경제", "문화", "미디어", "건강", "환경", "창업",
      "음식", "과학", "행사", "뷰티", "쇼핑", "인턴", "대회", "카페", "여행", "마케팅"
    ];

    const synonymMap = {
      "운동": "건강",
      "체육": "건강",
      "채용": "인턴"
    };
    let rawContent = response.data.choices[0].message.content;

    rawContent = rawContent.replace(/```json\s*([\s\S]*?)\s*```/, '$1').replace(/```([\s\S]*?)```/, '$1');

    let parsed;
    try {
        parsed = JSON.parse(rawContent);
    } catch (e) {
        console.error("❌ JSON 파싱 오류:", e.message);
        console.log("🔍 원본 응답:", rawContent);
        throw e;
    }

    // 카테고리 유효성 처리
    if (Array.isArray(parsed.categories)) {
        parsed.categories = parsed.categories
            .map(cat => synonymMap[cat] || cat) // 유사어 치환
            .filter(cat => validCategories.includes(cat)); // 허용된 항목만 유지

        if (parsed.categories.length === 0) {
            parsed.categories = ["교육"]; // 기본값
        }
    } else {
        parsed.categories = ["교육"]; //기본값  
    }

    return parsed;
};

module.exports = {classifyText};
