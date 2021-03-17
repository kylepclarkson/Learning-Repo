import React, { Component } from 'react'

import Input from './Input'

class FocusInput extends Component {

    clickHandler = () => {
        this.componentRef.current.focusInput()
    }

    render() {
        return (
            <div>
                {/* Get ref from parent component.  */}
                <Input ref={this.componentRef}/>
                <button onclick={this.clickHandler}>Focus Input</button>
            </div>
        )
    }
}

export default FocusInput
