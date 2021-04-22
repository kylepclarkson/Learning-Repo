import React, { Component } from 'react'

export class CountryPicker extends Component {

    constructor(props) {
        super(props);
        this.props = props
        this.handleSelect = this.handleSelect.bind(this);
    }

    handleSelect(event) {
        // pass region back to parent
        const value = event.target.value
        this.props.onRegionSelect(value)
    }

    render() {
        return (
            <select onChange={this.handleSelect} defaultValue="canda">
                <option value="canada">Canada</option>
                <option value="AB">Alberta</option>
                <option value="BC">British Columbia</option>
                <option value="MB">Manitoba</option>
                <option value="NB">New Brunswick</option>
                <option value="NL">Newfoundland and Labrador</option>
                <option value="NT">Northwest Territories</option>
                <option value="NS">Nova Scotia</option>
                <option value="NU">Nunavut</option>
                <option value="ON">Ontario</option>
                <option value="PE">Prince Edward Island</option>
                <option value="QC">Quebec</option>
                <option value="SK">Saskatchewan</option>
                <option value="YT">Yukon</option>
                <option value="RP">Repatriated</option>
            </select>
        )
    }
}

export default CountryPicker
