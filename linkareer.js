const axios = require('axios');
const cheerio = require('cheerio');
const xml2js = require('xml2js');
const {classifyText } =  require('./llm.js');

async function fetchsite() {
    const response = await axios.get("https://linkareer.com/sitemap/activities/114.xml");
    const parser = new xml2js.Parser();
    const sitemapData = await parser.parseStringPromise(response.data);
    const urls = sitemapData.urlset.url.map(url => url.loc[0]);
    const lasturls = urls.slice(-39, -30); //크롤링 개수 조절
    const result = [];
    for (const i of lasturls)
    {
        const data = await detailsite(i);
        result.push(data);
    }
    return result;
}

async function detailsite(link) {
    const response = await axios.get(link);
    const title = $('section h1').text();
    const $ = cheerio.load(response.data);
    const rawtext = $('.activity-detail-content').text();
    const cleantext = rawtext.replace(/<[^>]*>/g, '');
    const cleanedtext = cleantext.replace(/\s+/g, ' ').trim();
    const result = await classifyText(cleanedtext, title, link);
    return result;
}


module.exports = { fetchsite };