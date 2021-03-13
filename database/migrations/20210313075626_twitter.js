exports.up = (knex) => {
  return knex.schema
    .createTable("sources", (table) => {
      table.increments("id").primary();
      table.string("handle").unique();
      table.string("link");
      table.timestamps(true, true);
    })
    .createTable("tweets", (table) => {
      table.increments("id").primary();
      table
        .integer("source")
        .unsigned()
        .references("id")
        .inTable("sources")
        .onDelete("CASCADE")
        .index();
      table.string("tweet_id");
      table.string("tweet_date");
      table.timestamps(true, true);
    });
};

exports.down = (knex) => {
  return knex.schema.dropTableIfExists("tweets").dropTableIfExists("sources");
};
