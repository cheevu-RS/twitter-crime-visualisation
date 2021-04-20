import config from "../knexfile";
import knex, { Knex } from "knex";
import { Categories } from "./words";
import { Category, FilteredTweet, Tweet } from "./models";

async function go(builder: Knex.QueryBuilder) {
  return new Promise((resolve) => {
    builder.then((data) => resolve(data));
  });
}

function GetPercentageWords(t: string, words: string[]) {
  const tWords = t.replace(/[^a-zA-Z ]/g, "").split(" ");
  let count = 0;
  const found = [];
  for (const word of tWords) {
    const idx = words.findIndex((w) => w === word);
    if (idx >= 0) {
      found.push(word);
      ++count;
    }
  }
  const percentage = (count + 0.0) / (tWords.length + 0.0);
  return percentage * 100.0;
}

function FindCategory(t: string) {
  const tweet = t.toLowerCase();
  let mxPercentage = 0.0;
  let cat = "NonCrime";
  for (const [category, words] of Object.entries(Categories)) {
    const percentage = GetPercentageWords(tweet, words);
    if (percentage > mxPercentage) {
      mxPercentage = percentage;
      cat = category;
    }
  }
  return cat;
}

function LogCounts(tweets: any[]) {
  const count = new Map<string, number>();
  for (const [category, _] of Object.entries(Categories)) {
    count.set(category, 0);
  }

  for (const tweet of tweets) {
    const category = FindCategory(tweet.tweet);
    count.set(category, count.get(category)! + 1);
  }
  console.log(count);
}

async function InsertCategories(db: Knex) {
  const categories = Object.keys(Categories).map((cat, index) => {
    return {
      id: index + 1,
      category_name: cat,
    } as Category;
  });
  console.log(categories);
  await go(db("categories_two").insert(categories));
}

async function main(): Promise<void> {
  // @ts-ignore
  const db = knex(config.development);
  // await InsertCategories(db);

  // @ts-ignore
  const tweets: Tweet[] = await go(db.select("*").from("tweets"));
  // @ts-ignore
  const categories: Category[] = await go(db.select("*").from("categories_two"));

  console.log(categories);
  LogCounts(tweets);

  
  let done = 0;
  for (const tweet of tweets) {
    const category = FindCategory(tweet.tweet);
    if (category === "NonCrime") continue;
    const categoryId = categories.find((c) => c.category_name === category)!.id;
    const t = {
      ...tweet,
      category: categoryId,
    } as FilteredTweet;
    await go(db("filtered_tweets_two").insert(t));
    done += 1;
    if (done % 100 === 0) console.log('✅', done);
  }

}

main();
