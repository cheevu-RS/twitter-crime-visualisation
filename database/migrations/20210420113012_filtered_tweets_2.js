exports.up = (knex) => {
  return knex.schema
    .createTable("categories_two", (table) => {
      table.increments("id").primary();
      table.string("category_name").unique();
      table.timestamps(true, true);
    })
    .createTable("filtered_tweets_two", (table) => {
      table.increments("id").primary();
      table
        .integer("source")
        .unsigned()
        .references("id")
        .inTable("sources")
        .onDelete("CASCADE")
        .index();
      table
        .integer("category")
        .unsigned()
        .references("id")
        .inTable("categories_two")
        .onDelete("CASCADE")
        .index();
      table.string("tweet_id");
      table.string("tweet", 1024);
      table.string("tweet_date");
      table.timestamps(true, true);
      table.boolean("is_ok").defaultsTo(false);
    });
};

exports.down = (knex) => {
  return knex.schema
    .dropTableIfExists("filtered_tweets_two")
    .dropTableIfExists("categories_two")
};
