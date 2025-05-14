const pg = require("pg");

const { Pool } = pg;
const pool = new Pool({
  user: 'infotree',
  host: 'localhost',
  database: 'infotree',
  password: 'info1234',
  port: 5432,
});


// 저장 함수
async function savePostToBenefits(post) {
    const query = `
      I NSERT INTO benefits (
            title, start_date, end_date, description,
            owner_id, private, categories, channel_id,
            image, link, latitude, longitude
        )
        VALUES (
            $1, $2, $3, $4,
            $5, $6, $7, $8,
            $9, $10, $11, $12
        ) 
        ON CONFLICT (link) DO NOTHING;
    `;
    const startDate = post.start_date || new Date().toISOString().split('T')[0];
    const values = [
        post.title,
        startDate,
        post.end_date && post.end_date !== "" ? post.end_date : post.start_date,
        post.description,
        0,                    // owner_id (시스템 계정)
        false,                // private (공개)
        post.categories,    // TEXT[] 배열
        1,                    // channel_id
        post.image || null,
        post.link || null,
        null,                 // latitude
        null                  // longitude
    ];
  
    try {
      await pool.query(query, values);
      console.log(`✅ 저장됨: ${post.title}`);
    } catch (err) {
      console.error(`❌ 저장 오류: ${post.title}`, err.message);
    }
}  

// 전체 저장 처리
async function saveAllPosts(posts) {
    for (const post of posts) {
      await savePostToBenefits(post);
    }
}

module.exports = {saveAllPosts};