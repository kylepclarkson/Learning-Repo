import React, { Component } from 'react'

class RefsDemo extends Component {
    
    constructor(props) {
        super(props)
        this.inputRef = React.createRef()
        this.state = {
             
        }
    }
    
    componentDidMount() {
        // set focus to this item. 
        this.inputRef.current.focus()
        console.log(this.inputRef)
    }

    clickHandler = () => {
        alert(this.inputRef.current.value)
    }
    
    render() {
        return (
            <div>
                {/* make focus on this input. */}
                <input type="text" ref={this.inputRef}/>
                <button onClick={this.clickHandler}>Click</button>
            </div>
        )
    }
}

export default RefsDemo
