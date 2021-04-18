exports.up = (knex) => {
  return knex.schema
    .createTable("categories", (table) => {
      table.increments("id").primary();
      table.string("category_name").unique();
      table.timestamps(true, true);
    })
    .createTable("filtered_tweets", (table) => {
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
        .inTable("categories")
        .onDelete("CASCADE")
        .index();
      table.string("tweet_id");
      table.string("tweet", 1024);
      table.string("tweet_date");
      table.timestamps(true, true);
    });
};

exports.down = (knex) => {
  return knex.schema
    .dropTableIfExists("filtered_tweets")
    .dropTableIfExists("categories");
};
