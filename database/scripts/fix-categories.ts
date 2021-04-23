import config from "../knexfile";
import knex from "knex";
import { Category, FilteredTweet, Tweet } from "./models";
import { go } from "./utils";
import { Categories } from "./words";

const OldToNewCategoryNameMap = new Map<string, string>();

for (const [category, words] of Object.entries(Categories)) {
  OldToNewCategoryNameMap.set(category, category);
}

OldToNewCategoryNameMap.set("BikeTheft", "Theft");
OldToNewCategoryNameMap.set("Burglary", "Theft");
OldToNewCategoryNameMap.set("Robbery", "Theft");
OldToNewCategoryNameMap.set("Shoplifting", "Theft");
OldToNewCategoryNameMap.set("TheftFromThePerson", "Theft");

async function main() {
  const db = knex(config.development);
  // @ts-ignore
  const oldTweets: FilteredTweet[] = await go(
    db.select("*").from("filtered_tweets")
  );

  // @ts-ignore
  const newTweets: FilteredTweet[] = await go(
    db.select("*").from("filtered_tweets_two")
  );

  // @ts-ignore
  const oldCategories: Category[] = await go(db.select("*").from("categories"));

  // @ts-ignore
  const newCategories: Category[] = await go(
    db.select("*").from("categories_two")
  );

  const OldToNewCategoryIdMap = new Map<number, number>();
  for (const c1 of oldCategories) {
    for (const c2 of newCategories) {
      if (OldToNewCategoryNameMap.get(c1.category_name) == c2.category_name) {
        OldToNewCategoryIdMap.set(c1.id, c2.id);
      }
    }
  }

  console.log("OldToNewCategoryIdMap", OldToNewCategoryIdMap);

  let done = 0, okCnt = 0;
  for (const tweet of oldTweets) {
    if (tweet.is_ok) {
      let found = false;
      okCnt += 1;
      for (const t of newTweets) {
        if (t.tweet_id === tweet.tweet_id) {
          const query = db("filtered_tweets_two")
            .where({ id: t.id })
            .update({
              is_ok: true,
              category: OldToNewCategoryIdMap.get(tweet.category)!,
            });
          await go(query);
          done += 1;
          if (done % 25 === 0) console.log("✅", done);
          found = true;
          break;
        }
      }
      if (!found) {
        console.error(tweet);
      }
    }
  }

  console.log("✅", done);
  console.log("is_ok=true Count", okCnt);
}

main();
