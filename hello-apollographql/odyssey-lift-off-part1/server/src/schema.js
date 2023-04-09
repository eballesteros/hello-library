const { gql } = require("apollo-server");

const typeDefs = gql`

type Query{
    "Query to get tracks array"
    tracksForHome:[Track!]!
}

"A group of modules"
type Track{
    id: ID!
    title: String!
    author: Author!
    thumbnail: String
    lenght: Int
    modulesCount: Int
}

"Author of a track"
type Author{
    id: ID!
    name: String!
    photo: String
}
`

module.exports = typeDefs