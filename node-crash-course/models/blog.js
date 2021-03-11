const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// define schema
const blogSchema = new Schema({
    title: {
        type: String,
        required: true,
    },
    snippet: {
        type: String,
        required: true,
    },
    body: {
        type: String,
        required: true,
    }
}, { timestamps: true});

// define model based on blogSchema
const Blog = mongoose.model('Blog', blogSchema);

// export to use elsewhere
module.exports = Blog;

