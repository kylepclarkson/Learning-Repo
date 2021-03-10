const express = require('express');

// instace of express app.
const app = express();
// register view engine. Searches for 'views' directory by default.
app.set('view engine', 'ejs');

// listen for requests.
app.listen(3000);

app.get('/', (req, res) => {
    // auto-infers content type (sets header) and status code.
    // res.send('<p>Home Page</p>')
    // sendFile expects absolute path; need to set root to this directory.
    // res.sendFile('./views/index.html', { root: __dirname})

    const blogs = [
        {title: 'Machine Learning', snippet: 'How to code a computer to think.'},
        {title: 'Operations', snippet: 'The most for your buck.'},
        {title: 'Math', snippet: 'What you need to know.'},
    ]

    // render ejs
    res.render('index', { title: 'Home', blogs},);
});

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

    res.render('404')
})
