module.exports = {
  development: {
    client: "pg",
    connection: {
      host: "localhost",
      port: "5432",
      database: "fyp",
      user: "postgres",
      password: "postgres",
    },
    pool: {
      min: 0,
      max: 10,
    },
    migrations: {
      tableName: "knex_migrations",
    },
  },
};
