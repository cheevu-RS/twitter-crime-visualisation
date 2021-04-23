import config from "../knexfile";
import knex from "knex";
import { Category, FilteredTweet, Tweet } from "./models";
import { FindCategory, go, LogCounts } from "./utils";

async function main(): Promise<void> {
  const db = knex(config.development);
  // await InsertCategories(db);

  // @ts-ignore
  const tweets: Tweet[] = await go(db.select("*").from("tweets"));
  // @ts-ignore
  const categories: Category[] = await go(
    db.select("*").from("categories_two")
  );

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
    if (done % 100 === 0) console.log("✅", done);
  }
}

main();
