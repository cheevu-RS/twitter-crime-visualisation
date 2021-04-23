import { Knex } from "knex";
import { Categories } from "./words";
import { Category } from "./models";

export async function go(builder: Knex.QueryBuilder) {
  return new Promise((resolve) => {
    builder.then((data) => resolve(data));
  });
}

export function GetPercentageWords(t: string, words: string[]) {
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

export function FindCategory(t: string) {
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

export function LogCounts(tweets: any[]) {
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

export async function InsertCategories(db: Knex) {
  const categories = Object.keys(Categories).map((cat, index) => {
    return {
      id: index + 1,
      category_name: cat,
    } as Category;
  });
  console.log(categories);
  await go(db("categories_two").insert(categories));
}
