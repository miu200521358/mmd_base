import re
from struct import Struct

from mlib.base.base import Encoding
from mlib.base.math import MVector3D
from mlib.base.reader import BaseReader, StructUnpackType
from mlib.vmd.vmd_collection import VmdMotion
from mlib.vmd.vmd_part import (VmdBoneFrame, VmdCameraFrame, VmdIkOnOff,
                               VmdLightFrame, VmdMorphFrame, VmdShadowFrame,
                               VmdShowIkFrame)

RE_TEXT_TRIM = re.compile(rb"\x00+$")


class VmdReader(BaseReader[VmdMotion]):  # type: ignore
    def __init__(self) -> None:
        super().__init__()

    def create_model(self, path: str) -> VmdMotion:
        return VmdMotion(path=path)

    def read_by_buffer_header(self, motion: VmdMotion):
        self.define_encoding(Encoding.SHIFT_JIS)

        self.read_by_format[VmdBoneFrame] = StructUnpackType(
            self.read_bones,
            Struct("<I3f4f64B").unpack_from,
            4 + (4 * 3) + (4 * 4) + 64,
        )

        self.read_by_format[VmdMorphFrame] = StructUnpackType(
            self.read_morphs,
            Struct("<If").unpack_from,
            4 + 4,
        )

        self.read_by_format[VmdCameraFrame] = StructUnpackType(
            self.read_cameras,
            Struct("<I3f3f").unpack_from,
            4 + (4 * 3 * 2),
        )

        self.read_by_format[VmdLightFrame] = StructUnpackType(
            self.read_lights,
            Struct("<If3f3f24BIB").unpack_from,
            4 + 4 + (4 * 3 * 2) + 24 + 4 + 1,
        )

        self.read_by_format[VmdShadowFrame] = StructUnpackType(
            self.read_shadows,
            Struct("<IBf").unpack_from,
            4 + 1 + 4,
        )

        # vmdバージョン
        motion.signature = self.read_text(30)

        # モデル名
        motion.model_name = self.read_text(20)

    def read_by_buffer(self, motion: VmdMotion):
        # ボーンモーション
        self.read_bones(motion)

        # モーフモーション
        self.read_morphs(motion)

        # カメラ
        self.read_cameras(motion)

        # 照明
        self.read_lights(motion)

        # セルフ影
        self.read_shadows(motion)

        # セルフ影
        self.read_show_iks(motion)

    def define_read_text(self, encoding: Encoding):
        """
        テキストの解凍定義

        Parameters
        ----------
        encoding : Encoding
            デコードエンコード
        """

        def read_text(format_size: int) -> str:
            btext = self.unpack_text(format_size)
            # VMDは空白込みで入っているので、空白以降は削除する
            btext = RE_TEXT_TRIM.sub(b"", btext)

            return self.decode_text(encoding, btext)

        return read_text

    def read_bones(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            bf = VmdBoneFrame(register=True, read=True)

            bf.name = self.read_text(15)
            (
                bf.index,
                bf.position.x,
                bf.position.y,
                bf.position.z,
                bf.rotation.x,
                bf.rotation.y,
                bf.rotation.z,
                bf.rotation.scalar,
                bf.interpolations.translation_x.start.x,
                bf.interpolations.vals[1],
                bf.interpolations.vals[2],
                bf.interpolations.vals[3],
                bf.interpolations.translation_x.start.y,
                bf.interpolations.vals[5],
                bf.interpolations.vals[6],
                bf.interpolations.vals[7],
                bf.interpolations.translation_x.end.x,
                bf.interpolations.vals[9],
                bf.interpolations.vals[10],
                bf.interpolations.vals[11],
                bf.interpolations.translation_x.end.y,
                bf.interpolations.vals[13],
                bf.interpolations.vals[14],
                bf.interpolations.vals[15],
                bf.interpolations.translation_y.start.x,
                bf.interpolations.vals[17],
                bf.interpolations.vals[18],
                bf.interpolations.vals[19],
                bf.interpolations.translation_y.start.y,
                bf.interpolations.vals[21],
                bf.interpolations.vals[22],
                bf.interpolations.vals[23],
                bf.interpolations.translation_y.end.x,
                bf.interpolations.vals[25],
                bf.interpolations.vals[26],
                bf.interpolations.vals[27],
                bf.interpolations.translation_y.end.y,
                bf.interpolations.vals[29],
                bf.interpolations.vals[30],
                bf.interpolations.vals[31],
                bf.interpolations.translation_z.start.x,
                bf.interpolations.vals[33],
                bf.interpolations.vals[34],
                bf.interpolations.vals[35],
                bf.interpolations.translation_z.start.y,
                bf.interpolations.vals[37],
                bf.interpolations.vals[38],
                bf.interpolations.vals[39],
                bf.interpolations.translation_z.end.x,
                bf.interpolations.vals[41],
                bf.interpolations.vals[42],
                bf.interpolations.vals[43],
                bf.interpolations.translation_z.end.y,
                bf.interpolations.vals[45],
                bf.interpolations.vals[46],
                bf.interpolations.vals[47],
                bf.interpolations.rotation.start.x,
                bf.interpolations.vals[49],
                bf.interpolations.vals[50],
                bf.interpolations.vals[51],
                bf.interpolations.rotation.start.y,
                bf.interpolations.vals[53],
                bf.interpolations.vals[54],
                bf.interpolations.vals[55],
                bf.interpolations.rotation.end.x,
                bf.interpolations.vals[57],
                bf.interpolations.vals[58],
                bf.interpolations.vals[59],
                bf.interpolations.rotation.end.y,
                bf.interpolations.vals[61],
                bf.interpolations.vals[62],
                bf.interpolations.vals[63],
            ) = self.unpack(
                self.read_by_format[VmdBoneFrame].unpack,
                self.read_by_format[VmdBoneFrame].size,
            )

            motion.bones.append(bf)

    def read_morphs(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            mf = VmdMorphFrame(register=True, read=True)

            mf.name = self.read_text(15)
            (mf.index, mf.ratio,) = self.unpack(
                self.read_by_format[VmdMorphFrame].unpack,
                self.read_by_format[VmdMorphFrame].size,
            )

            motion.morphs.append(mf)

    def read_cameras(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            cf = VmdCameraFrame(register=True, read=True)
            degrees = MVector3D()

            (
                cf.index,
                cf.distance,
                cf.position.x,
                cf.position.y,
                cf.position.z,
                degrees.x,
                degrees.y,
                degrees.z,
                cf.interpolations.translation_x.start.x,
                cf.interpolations.translation_y.start.x,
                cf.interpolations.translation_z.start.x,
                cf.interpolations.rotation.start.x,
                cf.interpolations.distance.start.x,
                cf.interpolations.viewing_angle.start.x,
                cf.interpolations.translation_x.start.y,
                cf.interpolations.translation_y.start.y,
                cf.interpolations.translation_z.start.y,
                cf.interpolations.rotation.start.y,
                cf.interpolations.distance.start.y,
                cf.interpolations.viewing_angle.start.y,
                cf.interpolations.translation_x.end.x,
                cf.interpolations.translation_y.end.x,
                cf.interpolations.translation_z.end.x,
                cf.interpolations.rotation.end.x,
                cf.interpolations.distance.end.x,
                cf.interpolations.viewing_angle.end.x,
                cf.interpolations.translation_x.end.y,
                cf.interpolations.translation_y.end.y,
                cf.interpolations.translation_z.end.y,
                cf.interpolations.rotation.end.y,
                cf.interpolations.distance.end.y,
                cf.interpolations.viewing_angle.end.y,
            ) = self.unpack(
                self.read_by_format[VmdCameraFrame].unpack,
                self.read_by_format[VmdCameraFrame].size,
            )

            cf.rotation.degrees = degrees
            motion.cameras.append(cf)

    def read_lights(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            lf = VmdLightFrame(register=True, read=True)

            (
                lf.index,
                lf.color.x,
                lf.color.y,
                lf.color.z,
                lf.position.x,
                lf.position.y,
                lf.position.z,
            ) = self.unpack(
                self.read_by_format[VmdLightFrame].unpack,
                self.read_by_format[VmdLightFrame].size,
            )

            motion.lights.append(lf)

    def read_shadows(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            sf = VmdShadowFrame(register=True, read=True)

            (sf.index, sf.type, sf.distance,) = self.unpack(
                self.read_by_format[VmdShadowFrame].unpack,
                self.read_by_format[VmdShadowFrame].size,
            )

            motion.shadows.append(sf)

    def read_show_iks(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            kf = VmdShowIkFrame(register=True, read=True)
            kf.index = self.read_uint()
            kf.show = bool(self.read_byte())

            for _i in range(self.read_uint()):
                kf.iks.append(VmdIkOnOff(self.read_text(20), bool(self.read_byte())))

            motion.show_iks.append(kf)
