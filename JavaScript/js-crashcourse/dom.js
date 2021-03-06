console.log(window)

// Single element selection
// Select single elements. Will select the first elements.
// Get form
// console.log(document.getElementById('my-form'));
// const form = document.getElementById('my-form');
// console.log(form);

// Get container class
// console.log(document.querySelector('.container'))

// Multi element selection
// Select multiple elements. Return as a NodeList class. 
// console.log(document.querySelectorAll('.item'));

// 
const ul = document.querySelector('.items');

// Remove list from dom
// ul.remove();

// remove last item from list
// ul.lastElementChild.remove();

// update element
ul.firstElementChild.textContent = 'Hello';
ul.children[1].innerText = 'Brad';
ul.lastElementChild.innerHTML = '<h4>Night</h4>';

// 
const btn = document.querySelector('.btn');
btn.style.background = 'red';

// === events ===
btn.addEventListener('click', (e) => {
    e.preventDefault(); // disable form submission
    document.querySelector('#my-form').style.background = '#ccc';
    document.querySelector('body').classList.add('bg-dark');
})
