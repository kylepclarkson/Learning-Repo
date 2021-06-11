import React, { Component } from "react";
import { connect } from 'react-redux' // connects components to redux store. 
import { fetchPosts } from '../actions/postActions'
import PropTypes from 'prop-types'


class Posts extends Component {
  // constructor(props) {
  //   super(props);
  //   this.state = {
  //     posts: [], Now comes from redux. 
  //   };
  // }

  componentWillMount() {
    // Get posts using postsActions
    this.props.fetchPosts();
  }

  componentWillReceiveProps(nextProps) {
    if (nextProps.newPost) {
      this.props.posts.unshift(nextProps.newPost);
    }
  }

  render() {
    const postItems = this.props.posts.map(post => (
      <div key={post.id}>
        <h3>{post.title}</h3>
        <p>{post.body}</p>
      </div>
    ));

    return (
      <div>
        <h1>Posts</h1>
        {postItems}
      </div>
    );
  }
}

Posts.propTypes = {
  fetchPosts: PropTypes.func.isRequired,
  posts: PropTypes.array.isRequired,
  newPost: PropTypes.object
};

const mapStateToProps = state => ({
  posts: state.posts.items, // accessing 'posts' as defined in root reducer. 
  newPost: state.posts.item
})

// map redux state to state of component
export default connect(mapStateToProps, { fetchPosts } )(Posts);
