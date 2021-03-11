const express = require('express');
const morgan = require('morgan')
const mongoose = require('mongoose');
// const { result } = require('lodash');
const Blog = require('./models/blog');
const { render } = require('ejs');


// instace of express app.
const app = express();

// connect to mongoDB
const dbURI = 'mongodb+srv://db_admin:asdfasdf1234@nodecc.ucvd6.mongodb.net/note-tuts?retryWrites=true&w=majority';
mongoose.connect(dbURI, { useNewUrlParser: true, useUnifiedTopology: true})
    .then((result) => app.listen(3000))
    .catch((err) => console.log(err));
    // listen to requests only after connection is complete. 

// register view engine. Searches for 'views' directory by default.
app.set('view engine', 'ejs');

// === middleware ===
// serve static files
app.use(express.static('public'));
// accress form data
app.use(express.urlencoded({extended: true}));
// morgan - logging tool
app.use(morgan('dev'));



// === sandbox 

app.get('/add-blog', (req, res) => {
    const blog = new Blog({
        title: 'New Blog Post',
        snippet: 'About my new blog',
        body: 'Hello this is my new blog. It is dead simple ain\'t it?'
    });
    blog.save()
        .then((result) => {
            res.send(result);
        })
        .catch((err) => {
            console.log(err);
        })
})

app.get('/all-blogs', (req, res) => {
    Blog.find()
        .then((result) => {
            res.send(result);
        })
        .catch((err) => {
            console.log(err);
        })
});

app.get('/single-blog', (req, res) => {
    Blog.findById()
})


app.get('/', (req, res) => {
    // auto-infers content type (sets header) and status code.
    // res.send('<p>Home Page</p>')
    // sendFile expects absolute path; need to set root to this directory.
    // res.sendFile('./views/index.html', { root: __dirname})

    // const blogs = [
    //     {title: 'Machine Learning', snippet: 'How to code a computer to think.'},
    //     {title: 'Operations', snippet: 'The most for your buck.'},
    //     {title: 'Math', snippet: 'What you need to know.'},
    // ]

    // // render ejs
    // res.render('index', { title: 'Home', blogs},);
    res.redirect('/blogs')
});

/**
 * Create new blog using form data.
 */
app.post('/blogs', (req, res) => {
    const blog = new Blog(req.body)
    blog.save()
        .then((result) => {
            // redirect user to homepage
            res.redirect('/blogs');
        })
        .catch((err) => {
            console.log(err);
        })
})

/**
 * Get blog post by its id.
 */
app.get('/blogs/:id', (req, res) => {
    const id = req.params.id
    Blog.findById(id)
        .then(result => {
            res.render('details', {blog: result, title: 'Blog Details'})
        })
        .catch(err => {
            console.log(err)
        })
})

app.delete('/blogs/:id', (req, res) => {
    const id = req.params.id;
    Blog.findByIdAndDelete(id)
        .then(result => {
            // as front end is using AJAX, we cannot respond with a redirect.
            // respond with json pointing to where to go next.
            res.json({
                redirect: '/blogs'
            });
        })
        .catch(err => {
            console.log(err)
        })
})

/**
 * Display all blog posts.
 */
app.get('/blogs', (req, res) => {
    Blog.find().sort({createdAt: -1})
        .then((result) => {
            res.render('index', { title: 'All Blogs', blogs: result})
        })
        .catch((err) => {
            console.log(err)
        });
})

app.get('/about', (req, res) => {
    // auto-infers content type (sets header) and status code.
    // res.send('<p>About Page</p>')
    // res.sendFile('./views/about.html', { root: __dirname})

    res.render('about', {title: 'About'});
});

// redirect
app.get('/blogs/create', (req, res) => {
    res.render('create', {title: 'Blog Create'});
})

// 404
// use fires for all requests, only if the functions above have not been fired.
app.use((req, res) => {
    // res.status(404).sendFile('./views/404.html', { root: __dirname})

    res.render('404', {title: '404'})
})
