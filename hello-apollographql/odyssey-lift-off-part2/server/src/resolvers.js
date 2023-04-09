const resolvers = {
    Query: {
         // get all tracks, will be used to populate the homepage grid of our web client
        tracksForHome: (_, __, { dataSources }) => {
            return dataSources.trackAPI.getTraksForHome();
        },
    },
    Track: {
        // parent has info about the response from tracksForHome. Since tracksForHome returns a list
        // apollo server iterates though that list and calls the author resolver once per author
        author: ({ authorId }, _, { dataSources }) => {
            return dataSources.trackAPI.getAuthor(authorId)
        }
    }
}

module.exports = resolvers;