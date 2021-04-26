import axios from 'axios'

const url = 'https://api.opencovid.ca/'

export const summary = (region) => {

  axios.get(`${url}summary?loc=${region}`).then(res => {
    console.log('api response:', res)
    return res
  })
}