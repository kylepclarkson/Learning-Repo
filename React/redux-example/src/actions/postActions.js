/**
 *  The functions which are called and ran by the reducer.
 */

import { FETCH_POSTS, NEW_POST } from "./types";

// es6 arrow function
export const fetchPosts = () => (dispatch) => {
  // using thunk to execute function async (function in function using dispatch.)
  console.log("action: fetch posts.");
  fetch("https://jsonplaceholder.typicode.com/posts")
    .then((res) => res.json())
    .then((posts) =>
      // dispatch data to reducer
      dispatch({
        type: FETCH_POSTS,
        payload: posts,
      })
    );
};

export const createPost = (postData) => (dispatch) => {
  // using thunk to execute function async (function in function using dispatch.)
  console.log("action: create post.");
  fetch("https://jsonplaceholder.typicode.com/posts", {
    method: "POST",
    headers: {
      "Content-type": "application/json",
    },
    body: JSON.stringify(postData),
  })
    .then((res) => res.json())
    .then((post) =>
      dispatch({
        type: NEW_POST,
        payload: post,
      })
    );
};

// export function fetchPosts() {
//     // using thunk to execute function async (function in function using dispatch.)
//     return function (dispatch) {

//         fetch("https://jsonplaceholder.typicode.com/posts")
//         .then(res => res.json())
//         .then(posts => ({
//             // dispatch data to reducer
//             type: FETCH_POSTS,
//             payload: posts,
//         }));
//     }
// }
