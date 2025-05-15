const { fetchsite } = require("./linkareer.js");
const { saveAllPosts } = require("./pgdb.js");
const { crawl_club } = require("./campus_club.js");
const { crawl_activity } = require("./campus_activity.js");
const { crawl_contest } = require("./campus_contest.js");
const { krcrawl } = require("./koreasch.js");


async function crawl() {
    const number = 5; //크롤링 문서 개수 

    const clubresult = await crawl_club(number);
    await saveAllPosts(clubresult);
    //캠퍼스픽_동아리

    const activityresult = await crawl_activity(number);
    await saveAllPosts(activityresult);
    //캠퍼스픽_대외활동 

    const contestresult = await crawl_contest(number);
    await saveAllPosts(contestresult);
    //캠퍼스픽_공모전
     
    const linkresult = await fetchsite();
    await saveAllPosts(linkresult);
    //링커리어

    await krcrawl();
    //한국장학재단 공공데이터
}

( async () => {
    try {
        await crawl();
    } catch (err) {
        console.error(err);
    }
})();
