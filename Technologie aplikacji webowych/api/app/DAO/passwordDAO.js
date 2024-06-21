import mongoose from 'mongoose';
import * as _ from 'lodash';
import applicationException from '../service/applicationException';
import mongoConverter from '../service/mongoConverter';

const passwordSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'user', required: true, unique: true },
    password: { type: String, required: true }
  }, {
    collection: 'password_35196'
  });

const PasswordModel = mongoose.model('password_35196', passwordSchema);

async function createOrUpdate(data) {
  const existingUser = await PasswordModel.findOne({ userId: data.userId });
  if (existingUser) {
    existingUser.password = data.password;
    const result = await existingUser.save();
    return mongoConverter(result);
  } else {
    const result = await new PasswordModel({ userId: data.userId, password: data.password }).save();
    return mongoConverter(result);
  }
}

async function authorize(userId, password) {
  const result = await PasswordModel.findOne({ userId: userId, password: password });
  if (result && mongoConverter(result)) {
    return true;
  }
  throw applicationException.new(applicationException.UNAUTHORIZED, 'User and password does not match');
}

async function removeByUserId(userId) {
  return await PasswordModel.findOneAndRemove({ userId });
}

export default {
  createOrUpdate,
  authorize,
  removeByUserId,
  model: PasswordModel
};
