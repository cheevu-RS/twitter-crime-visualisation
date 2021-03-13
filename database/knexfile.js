module.exports = {
  development: {
    client: "pg",
    connection: {
      host: process.env.POSTGRES_HOST,
      database: "postgres",
      user: "postgres",
      password: process.env.POSTGRES_PASSWORD,
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
