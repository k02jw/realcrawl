const axios = require('axios');
const puppeteer = require('puppeteer');
const cheerio = require('cheerio');
const {classifyText}  =  require('./notitlellm.js');

async function detailsite(link) {

    const response = await axios.get(link, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.campuspick.com/',
        'Connection': 'keep-alive'
      },
      timeout: 50000
    });

    const $ = cheerio.load(response.data);
    const rawtext = $('body').text();
    const cleantext = rawtext.replace(/<[^>]*>/g, '');
    const cleanedtext = cleantext.replace(/\s+/g, ' ').trim();
    const result = await classifyText(cleanedtext, link);
    return result;
}

async function crawl_activity(num) {

    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    await page.goto('https://www.campuspick.com/activity', { waitUntil: 'networkidle2' });
    const content = await page.content();
    const result = [];
    let checker = new Set();
    
    const $ = cheerio.load(content);
    const element = $('a.item').toArray();
    let i = 0;
    for (const el of element)
    {
        i++;
        const href = $(el).attr('href');
        console.log(href);
        if (!checker.has(href))
        {
            const data = await detailsite(`https://www.campuspick.com${href}`);
            result.push(data);
            checker.add(href);
        }
        if (i == num)
        {
            break;
        }
    }
    await browser.close();
    return result;
}

module.exports = { crawl_activity };