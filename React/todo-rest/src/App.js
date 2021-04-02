import React from 'react'
import './App.css';

class App extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      todoList: [],
      // the item that will be updated
      activeItem: {
        id: null, title: "", completed: false
      },
      // if editing a post 
      editing: false,
    }

    // bind fetchTasks function to class.
    this.fetchTasks = this.fetchTasks.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.createCookie = this.createCookie.bind(this);
    this.startEdit = this.startEdit.bind(this);
    this.deleteItem = this.deleteItem.bind(this);
  }

  componentDidMount() {
    this.fetchTasks();
  }

  createCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      var cookies = document.cookie.split(';');
      for (var i=0; i<cookies.length; i++) {
        var cookie = cookies[i].trim();
        if (cookie.substring(0, name.length+1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  fetchTasks() {
    // fetch tasks from api.
    console.log('Fetching...');

    fetch('http://127.0.0.1:8000/api/task-list/')
      .then(res => res.json())
      .then(data =>
        this.setState({
          'todoList': data,
        })
      )
  }

  handleChange(e) {
    // update title of task entered in form.

    var name = e.target.name;
    var title = e.target.value;

    this.setState({
      activeItem: {
        ...this.state.activeItem,
        title: title
      }
    })
  }

  handleSubmit(e) {
    // submit form data. 
    e.preventDefault();
    console.log('Item: ', this.state.activeItem)
    // generate cookie token 
    var csrfToken = this.createCookie('csrftoken');

    // submit the active item to api
    var url = 'http://127.0.0.1:8000/api/task-create/'

    if (this.state.editing) {
      url = `http://127.0.0.1:8000/api/task-update/${this.state.id}/`
      this.setState({
        editing: false,
      })
    }

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-type': 'application/json',
        'X-CSRFToken': csrfToken,
      }, 
      body: JSON.stringify(this.state.activeItem)
    }).then((res) => {
      this.fetchTasks();
      this.setState({
        // clear item
        activeItem: {
          id: null, title: "", completed: false
        },
      })
    }).catch((err) => console.log("Error submiting data: ", err))
  }

  startEdit(task) {
    this.setState({
      activeItem: task,
      editing: true,
    })
  }

  deleteItem(task) {
    // generate cookie token 
    var csrfToken = this.createCookie('csrftoken');

    fetch(`http://127.0.0.1:8000/api/task-delete/${task.id}/`, {
      method: 'DELETE',
      headers: {
        'Content-type': 'application/json',
        'X-CSRFToken': csrfToken
      },
    }).then((res) => {
      this.fetchTasks();
    })
  }



  render() {
    var tasks = this.state.todoList
    var self = this

    return (
      <div className="container">
        <div id="task-container">
          <div id="form-wrapper">
            <form onSubmit={this.handleSubmit} id="id">
              <div className="flex-wrapper">
                <div style={{ flex: 6 }}>
                  <input type="text" onChange={this.handleChange} className="form-control" id="title" value={this.state.activeItem.title} name="title" placeholder="Add task" />
                </div>

                <div style={{ flex: 1 }}>
                  <input type="submit" className="btn btn-warning" id="submit" name="Add" />
                </div>

              </div>
            </form>
          </div>

          <div id="list-wrapper">
            {/* display tasks */}
            {tasks.map(function (task, index) {
              return (
                <div
                  key={index}
                  className="task-wrapper flex-wrapper">
                  <div style={{flex: 7}}>
                    <span>{task.title}</span>
                  </div>
                  
                  <div style={{flex: 1}}>
                    <button onClick={() => self.startEdit(task)} className="btn btn-sm btn-outline-info">Edit</button>
                  </div>
                  
                  <div style={{flex: 1}}>
                    <button onClick={() => self.deleteItem(task)} className="btn btn-sm btn-outline-dark">Remove</button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    )
  }
}

export default App;
