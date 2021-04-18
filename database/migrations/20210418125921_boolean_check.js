exports.up = (knex) => {
  return knex.schema
    .table("filtered_tweets", (table) => {
      table.boolean("is_ok").defaultsTo(true);
    })
    .then(() => {
      return knex.schema.raw("UPDATE filtered_tweets SET is_ok = false");
    });
};

exports.down = (knex) => {
  return knex.schema.table("filtered_tweets", (table) => {
    table.dropColumn("is_ok");
  });
};
