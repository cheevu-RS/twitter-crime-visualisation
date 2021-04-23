import knex from "knex";
import config from "../knexfile";
import { Category, FilteredTweet } from "./models";
import { go, LogCounts } from "./utils";

async function main() {
  const db = knex(config.development);

  // @ts-ignore
  const tweets: FilteredTweet[] = await go(
    db.select("*").from("filtered_tweets_two").where({ is_ok: true })
  );

  // @ts-ignore
  const categories: Category[] = await go(
    db.select("*").from("categories_two")
  );

  const count = new Map<string, number>();
  const categoryIdToName = new Map<number, string>();
  for (const category of categories) {
    count.set(category.category_name, 0);
    categoryIdToName.set(category.id, category.category_name);
  }

  for (const tweet of tweets) {
    const key = categoryIdToName.get(tweet.category)!;
    count.set(key, 1 + count.get(key)!);
  }
  console.log(count);
}

main();
