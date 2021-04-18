export default {
  development: {
    client: "pg",
    connection: {
      host: "127.0.0.1",
      port: "7000",
      database: "fyp",
      user: "postgres",
      password: "password",
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
